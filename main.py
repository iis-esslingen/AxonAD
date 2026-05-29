import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import torch
import gc
import re
import random

from TSB_AD.model_wrapper import (
    run_Unsupervise_AD,
    run_Semisupervise_AD,
    Unsupervise_AD_Pool,
    Semisupervise_AD_Pool,
)
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict

warnings.filterwarnings("ignore")

def set_random_seed(seed=42):
    """Set seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch deterministic (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 2024

# ------------------------------------------------------------------
# Configuration
# Root reproduction entry point for the paper tables/CSV outputs.
# Edit AD_NAMES, BENCHMARK, SPLIT_MODE, and DATA_ROOTS before running.
# ------------------------------------------------------------------
AD_NAMES = [
        # Unsupervised Models (Multivariate-capable)
    'IForest', 'LOF', 'HBOS', 'KNN', 'KMeansAD', 
    'COPOD', 'CBLOF', 'EIF', 'RobustPCA',
    
    # Semi-supervised Deep Learning Models (Multivariate)
    'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'USAD', 'OmniAnomaly',
    'AnomalyTransformer', 'TimesNet', 'FITS', 'Donut', 'MatrixProfile', 'AxonAD', 'StreamVAE',
    
    # NEW VAE-based Models (Multivariate)
    'WVAE', 'VSVAE', 'VASP', 'SISVAE', 'MAVAE',
    
    # NEW Transformer/Graph Models (Multivariate)
    'TFTResidual', 'GDN',

    # Slow Models
    'PCA', 'COF',  'OFA', 
    'M2N2',

    # Kernel/Distribution Models
]

# Which benchmark? "M" (multivariate, TSB-AD-M), "U" (univariate, TSB-AD-U), or "UCR"
# For the paper benchmark, use "M" or "U".
BENCHMARK = "UCR"

# Which official split to use:
#   For M: "eval", "tuning", "telemetry", or "all"
#   For U: "eval", "eval_full", "tuning", or "all"
#   For UCR: ignored (all 250 series are used)
SPLIT_MODE = "eval"

# Data roots – adjust if needed
if BENCHMARK.upper() == "M":
    DATA_ROOTS = ["Datasets/TSB-AD-M"]
elif BENCHMARK.upper() == "UCR":
    DATA_ROOTS = ["/Users/KAOEZER/Downloads/AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData"]
else:
    DATA_ROOTS = ["Datasets/TSB-AD-U"]

FILE_LIST_DIR = "Datasets/File_List"
output_file = f"tsb_ad_{BENCHMARK.lower()}_{SPLIT_MODE}_results.csv"

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
def list_all_csvs(roots):
    files = []
    for r in roots:
        files.extend(sorted(glob.glob(os.path.join(r, "*.csv"))))
    return files


def list_all_ucr_txts(roots):
    """Discover all UCR .txt files under the given root directories."""
    files = []
    for r in roots:
        files.extend(sorted(glob.glob(os.path.join(r, "*.txt"))))
    return files


def parse_ucr_filename(filename):
    """
    Parse train_end, anomaly_start, anomaly_end from a UCR filename.
    Format: {id}_UCR_Anomaly_{name}_{train_end}_{anomaly_start}_{anomaly_end}.txt
    Returns (train_end, anomaly_start, anomaly_end) as ints, or None on failure.
    """
    basename = os.path.basename(filename)
    match = re.search(r"_(\d+)_(\d+)_(\d+)\.txt$", basename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None


def load_ucr_txt(filepath):
    """
    Load a UCR anomaly dataset .txt file.
    Returns X (numpy array, shape [N, 1]) and y (binary label array, shape [N]).
    Labels are derived from the filename: anomaly_start..anomaly_end (0-based, inclusive).
    """
    basename = os.path.basename(filepath)
    parsed = parse_ucr_filename(basename)
    if parsed is None:
        raise ValueError(f"Cannot parse UCR filename: {basename}")
    train_end, anomaly_start, anomaly_end = parsed

    values = np.loadtxt(filepath)
    X = values.reshape(-1, 1).astype(float)
    n = len(X)

    y = np.zeros(n, dtype=int)
    # Filename indices are 1-based; clamp to valid range
    a_start = max(0, anomaly_start - 1)
    a_end = min(n - 1, anomaly_end - 1)
    y[a_start : a_end + 1] = 1

    return X, y, train_end


def parse_train_index(filename):
    """
    Safely extracts the training index from filenames like:
    '001_..._tr_2500_...csv'
    Returns None if not found.
    """
    match = re.search(r"_tr_(\d+)_", filename)
    if match:
        return int(match.group(1))
    return None


def _read_filelist_csv(path):
    """
    Read a single TSB-AD file-list CSV and return a set of basenames
    (e.g., '001_NAB_id_1_Facility_tr_1007_1st_2014.csv').
    """
    if not os.path.exists(path):
        print(f"[WARN] File list not found: {path}")
        return set()

    df = pd.read_csv(path)

    # Try to guess the column that holds the file/path
    fname_col = None
    candidates = ["data_direc", "data_dir", "file", "filename", "path", "ts_name"]
    for cand in candidates:
        for col in df.columns:
            col_lower = col.lower()
            if col_lower == cand or cand in col_lower:
                fname_col = col
                break
        if fname_col is not None:
            break

    # Fallback: if we still don't know, assume first column
    if fname_col is None:
        fname_col = df.columns[0]

    return set(df[fname_col].astype(str).apply(os.path.basename))


def load_official_split_filenames(benchmark="M", split_mode="eval"):
    """
    Map (benchmark, split_mode) -> which File_List CSVs to use,
    and return the set of allowed basenames.

    For BENCHMARK="M":
      - eval   -> TSB-AD-M.csv
      - tuning -> TSB-AD-M-Tuning.csv
      - all    -> union of the above

    For BENCHMARK="U":
      - eval       -> TSB-AD-U-Eva.csv        (main curated eval)
      - eval_full  -> TSB-AD-U-Eva-Full.csv   (larger eval set)
      - tuning     -> TSB-AD-U-Tuning.csv
      - all        -> TSB-AD-U.csv            (all univariate)
    """
    benchmark = benchmark.upper()
    split_mode = split_mode.lower()

    list_paths = []

    if benchmark == "M":
        if split_mode in ("eval", "all"):
            list_paths.append(os.path.join(FILE_LIST_DIR, "TSB-AD-M-Eva.csv"))
        if split_mode in ("tuning", "all"):
            list_paths.append(os.path.join(FILE_LIST_DIR, "TSB-AD-M-Tuning.csv"))
        if split_mode == "telemetry":
            list_paths.append(os.path.join(FILE_LIST_DIR, "TSB-AD-M-Telemetry.csv"))
    elif benchmark == "U":
        if split_mode == "eval":
            list_paths.append(os.path.join(FILE_LIST_DIR, "TSB-AD-U-Eva.csv"))
        elif split_mode == "eval_full":
            list_paths.append(os.path.join(FILE_LIST_DIR, "TSB-AD-U-Eva-Full.csv"))
        elif split_mode == "tuning":
            list_paths.append(os.path.join(FILE_LIST_DIR, "TSB-AD-U-Tuning.csv"))
        elif split_mode == "all":
            list_paths.append(os.path.join(FILE_LIST_DIR, "TSB-AD-U.csv"))
    elif benchmark == "UCR":
        # UCR has no separate file lists; always use all 250 series
        return None
    else:
        raise ValueError("BENCHMARK must be 'M', 'U', or 'UCR'")

    if not list_paths:
        print(f"[WARN] No list_paths for (benchmark={benchmark}, split_mode={split_mode}).")
        return None

    allowed = set()
    for p in list_paths:
        allowed |= _read_filelist_csv(p)

    if not allowed:
        print("[WARN] No filenames loaded from official split file lists; using all datasets.")
        return None

    print(f"[INFO] Loaded {len(allowed)} filenames from official {benchmark}-{split_mode} file list(s).")
    return allowed


# ------------------------------------------------------------------
# Execution Logic
# ------------------------------------------------------------------
IS_UCR = BENCHMARK.upper() == "UCR"

# 1) Load official split lists (if available; not used for UCR)
allowed_filenames = load_official_split_filenames(BENCHMARK, SPLIT_MODE)

# 2) Discover all dataset files under the configured roots
if IS_UCR:
    all_datasets = list_all_ucr_txts(DATA_ROOTS)
else:
    all_datasets = list_all_csvs(DATA_ROOTS)
print(f"Found {len(all_datasets)} dataset files before split filtering")

# 3) Apply split filtering if we have a valid file list
if allowed_filenames is not None:
    filtered = [
        p for p in all_datasets
        if os.path.basename(p) in allowed_filenames
    ]
    print(
        f"{len(filtered)} dataset files after applying official "
        f"'{BENCHMARK}-{SPLIT_MODE}' split filtering"
    )
    all_datasets = filtered

print(f"Final dataset count: {len(all_datasets)}")

# Load existing results (to allow resuming)
if os.path.exists(output_file):
    try:
        # Check if file has content
        if os.path.getsize(output_file) > 0:
            existing_df = pd.read_csv(output_file)
            if len(existing_df) > 0:
                results = existing_df.to_dict("records")
                completed_pairs = set(zip(existing_df["AD_Name"], existing_df["dataset"]))
                print(f"Resuming: Found {len(completed_pairs)} completed (algorithm, dataset) pairs")
            else:
                results = []
                completed_pairs = set()
        else:
            # File is empty, start fresh
            results = []
            completed_pairs = set()
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"Warning: Could not read existing results file ({e}). Starting fresh.")
        results = []
        completed_pairs = set()
else:
    results = []
    completed_pairs = set()

for ad_name in AD_NAMES:
    set_random_seed(SEED)
    print(f"\n{'=' * 60}")
    print(f"Running algorithm: {ad_name}")
    print(f"{'=' * 60}")

    algo_results = []
    optimal_hp = Optimal_Multi_algo_HP_dict.get(ad_name, {})

    for data_path in tqdm(all_datasets, desc=f"AD={ad_name}"):
        filename = os.path.basename(data_path)

        # Skip if this (algorithm, dataset) is already done
        if (ad_name, filename) in completed_pairs:
            continue

        train_idx = 0  # default for unsupervised
        X_train = None
        X_test = None

        try:
            # 1. Data Loading
            if IS_UCR:
                X, y, ucr_train_end = load_ucr_txt(data_path)
            else:
                df = pd.read_csv(data_path).dropna()
                # Features: all except last column (Label)
                X = df.iloc[:, :-1].values.astype(float)
                y = df.iloc[:, -1].astype(int).to_numpy()
                df = None

            # 2. Sliding Window (Periodicity)
            if X.ndim > 1 and X.shape[1] > 1:
                slidingWindow = find_length_rank(X[:, 0].reshape(-1, 1), rank=1)
            else:
                slidingWindow = find_length_rank(X, rank=1)

            # 3. Run Detection
            if ad_name in Semisupervise_AD_Pool:
                if IS_UCR:
                    train_idx = ucr_train_end
                else:
                    train_idx = parse_train_index(filename)
                    if train_idx is None:
                        train_idx = int(len(X) * 0.3)

                X_train = X[:train_idx, :]
                X_test = X

                scores = run_Semisupervise_AD(ad_name, X_train, X_test, **optimal_hp)
            else:
                scores = run_Unsupervise_AD(ad_name, X, **optimal_hp)

            # 4. Metrics
            metrics = get_metrics(scores, y, slidingWindow=slidingWindow)

            row = {
                "AD_Name": ad_name,
                "dataset": filename,
                "data_len": len(X),
                "train_split": train_idx,
                "benchmark": BENCHMARK,
                "split_mode": SPLIT_MODE,
            }
            row.update(metrics)
            algo_results.append(row)

        except KeyboardInterrupt:
            print("Interrupted by user. Saving and exiting...")
            pd.DataFrame(results + algo_results).to_csv(output_file, index=False)
            raise

        except Exception as e:
            print(f"\n[ERROR] {ad_name} failed on {filename}: {str(e)}")

            # Optional: record a 'failure row' with zero metrics
            zero_metrics = {
                k: 0.0 for k in [
                    "AUC-PR", "AUC-ROC", "VUS-PR", "VUS-ROC",
                    "Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1",
                    "Affiliation-F"  # adapt keys to your get_metrics output
                ]
            }

            row = {
                "AD_Name": ad_name,
                "dataset": filename,
                "data_len": len(X) if "X" in locals() else 0,
                "train_split": train_idx,
                "benchmark": BENCHMARK,
                "split_mode": SPLIT_MODE,
            }
            row.update(zero_metrics)
            algo_results.append(row)


        finally:
            # 5. Memory Cleanup (important for all models, especially PCA/LOF/KNN with sliding windows)
            if scores is not None:
                del scores
            if X is not None:
                del X
            if y is not None:
                del y
            if not IS_UCR and "df" in locals() and df is not None:
                del df
            if X_train is not None:
                del X_train
            if X_test is not None:
                del X_test
            gc.collect()  # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Save after every Algorithm completes
    results.extend(algo_results)
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"✓ Saved results for {ad_name}")
