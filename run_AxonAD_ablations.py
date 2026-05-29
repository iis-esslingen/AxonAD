#!/usr/bin/env python3
"""Run AxonAD ablations and save results as a CSV compatible with aggregate_results.py.

This script mirrors the repo-root main.py style (dataset discovery + split filtering + metrics)
but runs a set of AxonAD ablation variants.

Output CSV columns match the expected schema:
  AD_Name, dataset, data_len, train_split, benchmark, split_mode, <metric columns...>

Ablation variants are encoded into AD_Name (e.g., AxonAD__base, AxonAD__mask02)
so aggregate_results.py groups them correctly without needing schema changes.
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import random
import re
import warnings
from dataclasses import dataclass
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import math

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict, Optimal_Uni_algo_HP_dict
from TSB_AD.models.AxonAD import TCNTransformer
from TSB_AD.models.AxonAD import TCNQueryPredictor
from TSB_AD.utils.dataset import ReconstructDataset


warnings.filterwarnings("ignore")


def set_random_seed(seed: int = 2024) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def list_all_csvs(roots: list[str]) -> list[str]:
	files: list[str] = []
	for root in roots:
		files.extend(sorted(glob.glob(os.path.join(root, "*.csv"))))
	return files


def parse_train_index(filename: str) -> int | None:
	match = re.search(r"_tr_(\d+)_", filename)
	if match:
		return int(match.group(1))
	return None


def _read_filelist_csv(path: str) -> set[str]:
	if not os.path.exists(path):
		print(f"[WARN] File list not found: {path}")
		return set()

	df = pd.read_csv(path)

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

	if fname_col is None:
		fname_col = df.columns[0]

	return set(df[fname_col].astype(str).apply(os.path.basename))


def load_official_split_filenames(file_list_dir: str, benchmark: str, split_mode: str) -> set[str] | None:
	benchmark = benchmark.upper()
	split_mode = split_mode.lower()

	list_paths: list[str] = []

	if benchmark == "M":
		if split_mode in ("eval", "all"):
			list_paths.append(os.path.join(file_list_dir, "TSB-AD-M-Eva.csv"))
		if split_mode in ("tuning", "all"):
			list_paths.append(os.path.join(file_list_dir, "TSB-AD-M-Tuning.csv"))
		if split_mode == "telemetry":
			list_paths.append(os.path.join(file_list_dir, "TSB-AD-M-Telemetry.csv"))
	elif benchmark == "U":
		if split_mode == "eval":
			list_paths.append(os.path.join(file_list_dir, "TSB-AD-U-Eva.csv"))
		elif split_mode == "eval_full":
			list_paths.append(os.path.join(file_list_dir, "TSB-AD-U-Eva-Full.csv"))
		elif split_mode == "tuning":
			list_paths.append(os.path.join(file_list_dir, "TSB-AD-U-Tuning.csv"))
		elif split_mode == "all":
			list_paths.append(os.path.join(file_list_dir, "TSB-AD-U.csv"))
	else:
		raise ValueError("BENCHMARK must be 'M' or 'U'")

	if not list_paths:
		print(f"[WARN] No list_paths for (benchmark={benchmark}, split_mode={split_mode}).")
		return None

	allowed: set[str] = set()
	for p in list_paths:
		allowed |= _read_filelist_csv(p)

	if not allowed:
		print("[WARN] No filenames loaded from official split file lists; using all datasets.")
		return None

	print(f"[INFO] Loaded {len(allowed)} filenames from official {benchmark}-{split_mode} file list(s).")
	return allowed


@dataclass(frozen=True)
class AblationVariant:
	"""A single ablation configuration."""

	name: str
	constructor_overrides: dict
	model_overrides: dict


class TCNTransformerAblation(TCNTransformer):
	"""Script-local wrapper enabling training/scoring ablations without changing library files."""

	def __init__(
		self,
		*,
		train_objective: str = "balanced",  # balanced|rec|jepa|pred_h|pred_k|pred_v|pred_attn_q|pred_attn_qk
		score_mode: str = "mse+jepa",  # mse|mse+jepa|mse+kl|mse+jepa+kl
		**kwargs,
	):
		super().__init__(**kwargs)
		self._train_objective = train_objective
		self._score_mode = score_mode
		self._mse_loss = nn.MSELoss()

		# Optional extra predictors for key/value objectives.
		# Rebuild optimizer if we add new trainable modules.
		objective = str(train_objective).lower().strip()
		self.k_pred_net = None
		self.v_pred_net = None
		if objective in {"pred_k", "pred_attn_qk"}:
			self.k_pred_net = TCNQueryPredictor(d_model=self.model.d_model).to(self.device)
		if objective == "pred_v":
			self.v_pred_net = TCNQueryPredictor(d_model=self.model.d_model).to(self.device)
		if self.k_pred_net is not None or self.v_pred_net is not None:
			# NOTE: This wrapper is not an nn.Module; don't use self.parameters().
			params = list(self.model.parameters())
			if self.k_pred_net is not None:
				params += list(self.k_pred_net.parameters())
			if self.v_pred_net is not None:
				params += list(self.v_pred_net.parameters())
			self.opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-5)

	def _shift_history(self, h_detached: torch.Tensor, s: int) -> torch.Tensor:
		"""Create the same history-only shift used by the built-in Q predictor."""
		B, T, C = h_detached.shape
		s = max(0, min(int(s), T))
		if s == 0:
			return h_detached
		x_shift = h_detached.new_zeros(B, T, C)
		x_shift[:, s:, :] = h_detached[:, :-s, :]
		return x_shift

	def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
		"""[B,H,T,D] -> [B,T,H*D]"""
		return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], -1)

	def _cos_masked_loss_flat(self, pred_flat: torch.Tensor, tgt_flat: torch.Tensor, tmask: torch.Tensor) -> torch.Tensor:
		"""Cosine distance averaged over masked timesteps; flat tensors are [B,T,C]."""
		B, T, _ = pred_flat.shape
		if tmask is None or tmask.numel() == 0:
			return pred_flat.new_zeros(())
		m = tmask.to(pred_flat.device)
		if m.dtype != torch.bool:
			m = m.bool()
		if m.sum() == 0:
			return pred_flat.new_zeros(())

		p = F.normalize(pred_flat, dim=-1)
		t = F.normalize(tgt_flat, dim=-1)
		cos = (p * t).sum(dim=-1)  # [B,T]
		dist = 1.0 - cos
		return dist[m].mean()

	def _attn_logits(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
		"""Compute attention logits: [B,H,T,D] x [B,H,T,D] -> [B,H,T,T]."""
		D = q.shape[-1]
		return (q @ k.transpose(-2, -1)) / math.sqrt(D)

	def _attn_kl_masked(self, logits_teacher: torch.Tensor, logits_student: torch.Tensor, tmask: torch.Tensor) -> torch.Tensor:
		"""KL( teacher || student ) over attention distributions, averaged on masked query timesteps."""
		# logits: [B,H,T,T]
		if tmask is None or tmask.numel() == 0:
			return logits_teacher.new_zeros(())
		B, H, T, _ = logits_teacher.shape
		m = tmask.to(logits_teacher.device)
		if m.dtype != torch.bool:
			m = m.bool()
		if m.sum() == 0:
			return logits_teacher.new_zeros(())

		# Mask out future keys according to forecast shift s inside tmask already (tmask disables <s).
		# We keep full key range here; fairness ablations can extend this later.
		logp = F.log_softmax(logits_teacher, dim=-1)
		logq = F.log_softmax(logits_student, dim=-1)
		p = logp.exp()
		kl = (p * (logp - logq)).sum(dim=-1)  # [B,H,T]
		m3 = m.view(B, 1, T).expand(B, H, T)
		return kl[m3].mean()

	def fit(self, data):
		split = int((1 - self.validation_size) * len(data))
		if split <= self.win_size:
			tsTrain = data
			tsValid = data[:0]
		else:
			tsTrain = data[:split]
			tsValid = data[split:]

		train_loader = DataLoader(
			ReconstructDataset(tsTrain, window_size=self.win_size),
			batch_size=self.batch_size,
			shuffle=True,
		)
		valid_loader = DataLoader(
			ReconstructDataset(tsValid, window_size=self.win_size),
			batch_size=self.batch_size,
			shuffle=False,
		)

		train_objective = str(self._train_objective).lower().strip()
		allowed_obj = {
			"balanced",
			"rec",
			"jepa",  # predicts query-representation (Q) via built-in JEPA loss
			"pred_h",  # predict hidden state h_norm
			"pred_k",  # predict keys K
			"pred_v",  # predict values V
			"pred_attn_q",  # match attention map using predicted Q + teacher K
			"pred_attn_qk",  # match attention map using predicted Q + predicted K
		}
		if train_objective not in allowed_obj:
			raise ValueError(f"train_objective must be one of: {sorted(allowed_obj)}")

		for epoch in range(1, self.epochs + 1):
			self.model.train()
			loop = tqdm.tqdm(train_loader, leave=True)

			for d, _ in loop:
				d = d.to(self.device)
				rec, q_real, q_pred, k, q_tgt, tmask = self.model(d)

				# Recompute encoder hidden for additional objectives
				h = self.model.embed(d) + self.model.pos
				h_norm = self.model.ln1(h)
				h_det = h_norm.detach()
				s = max(0, min(int(self.model.forecast_steps), h_det.shape[1]))
				x_shift = self._shift_history(h_det, s)

				# Predictors in flat space [B,T,C]
				q_pred_flat = self.model.attn.q_pred_net(x_shift)
				k_tgt = k.detach()
				v_tgt = self.model.attn._split_heads(self.model.attn.wv(h_norm)).detach()

				L_rec = self._mse_loss(rec, d)
				L_jepa = self._jepa_masked_loss(q_tgt, q_pred, tmask)

				# Additional objectives
				if train_objective == "pred_h":
					L_aux = self._cos_masked_loss_flat(q_pred_flat, h_det, tmask)
				elif train_objective == "pred_k":
					if self.k_pred_net is None:
						raise RuntimeError("k_pred_net not initialized")
					k_pred_flat = self.k_pred_net(x_shift)
					k_pred = self.model._split_heads(k_pred_flat, self.model.attn.num_heads)
					k_tgt_flat = self._merge_heads(k_tgt)
					L_aux = self._cos_masked_loss_flat(self._merge_heads(k_pred), k_tgt_flat, tmask)
				elif train_objective == "pred_v":
					if self.v_pred_net is None:
						raise RuntimeError("v_pred_net not initialized")
					v_pred_flat = self.v_pred_net(x_shift)
					v_pred = self.model._split_heads(v_pred_flat, self.model.attn.num_heads)
					v_tgt_flat = self._merge_heads(v_tgt.detach())
					L_aux = self._cos_masked_loss_flat(self._merge_heads(v_pred), v_tgt_flat, tmask)
				elif train_objective == "pred_attn_q":
					q_pred = self.model._split_heads(q_pred_flat, self.model.attn.num_heads)
					logits_t = self._attn_logits(q_real.detach(), k_tgt)
					logits_s = self._attn_logits(q_pred, k_tgt)
					L_aux = self._attn_kl_masked(logits_t, logits_s, tmask)
				elif train_objective == "pred_attn_qk":
					if self.k_pred_net is None:
						raise RuntimeError("k_pred_net not initialized")
					k_pred_flat = self.k_pred_net(x_shift)
					q_pred = self.model._split_heads(q_pred_flat, self.model.attn.num_heads)
					k_pred = self.model._split_heads(k_pred_flat, self.model.attn.num_heads)
					logits_t = self._attn_logits(q_real.detach(), k_tgt)
					logits_s = self._attn_logits(q_pred, k_pred)
					L_aux = self._attn_kl_masked(logits_t, logits_s, tmask)
				else:
					L_aux = None

				if train_objective == "rec":
					loss = L_rec
				elif train_objective == "jepa":
					loss = L_jepa
				elif train_objective.startswith("pred_"):
					loss = L_aux
				else:
					s_rec = self.model.s_rec
					s_kl = self.model.s_kl
					loss = torch.exp(-s_rec) * L_rec + s_rec + torch.exp(-s_kl) * L_jepa + s_kl

				self.opt.zero_grad(set_to_none=True)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
				self.opt.step()

				self.model.ema_update_target()

				with torch.no_grad():
					tail_jepa = self._jepa_tail_dist(q_tgt, q_pred).mean().item()
					tail_kl = self._kl_tail(q_real, q_pred, k).mean().item()

				loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
				loop.set_postfix(
					rec=float(L_rec.item()),
					jepa=float(L_jepa.item()),
					tail_jepa=float(tail_jepa),
					tail_kl=float(tail_kl),
					tau=float(self.model.tau().item()),
					loss=float(loss.item()),
				)

			if len(valid_loader) > 0:
				self.model.eval()
				val_rec_sum, n_batches = 0.0, 0

				with torch.no_grad():
					for d, _ in valid_loader:
						d = d.to(self.device)
						rec, *_ = self.model(d)
						v_rec = self._mse_loss(rec, d)
						val_rec_sum += float(v_rec.item())
						n_batches += 1

				val_rec_avg = val_rec_sum / max(1, n_batches)
				self.early_stopping(val_rec_avg, self.model)
				print(f"[VAL] rec={val_rec_avg:.6f}")

				if self.early_stopping.early_stop:
					print("Early stopping triggered")
					break

		self._calibrate(tsTrain)

	def decision_function(self, data):
		loader = DataLoader(
			ReconstructDataset(data, window_size=self.win_size),
			batch_size=self.batch_size,
			shuffle=False,
		)

		mode = str(self._score_mode).lower().replace(" ", "")
		allowed = {"mse", "mse+jepa", "mse+kl", "mse+jepa+kl"}
		if mode not in allowed:
			raise ValueError(f"score_mode must be one of: {sorted(allowed)}")

		self.model.eval()
		scores = []
		eps = 1e-9

		with torch.inference_mode():
			for d, _ in tqdm.tqdm(loader, leave=True):
				d = d.to(self.device)
				rec, q_real, q_pred, k, q_tgt, _ = self.model(d)

				mse_rec = (rec - d).pow(2).sum(dim=-1).mean(dim=1)
				jepa = self._jepa_tail_dist(q_tgt, q_pred)
				kl = self._kl_tail(q_real, q_pred, k)

				z_mse = (mse_rec - self.mse_med) / (self.mse_iqr + eps)
				z_jepa = (jepa - self.jepa_med) / (self.jepa_iqr + eps)
				z_kl = (kl - self.kl_med) / (self.kl_iqr + eps)

				if mode == "mse":
					score = z_mse
				elif mode == "mse+kl":
					score = z_mse + z_kl
				elif mode == "mse+jepa+kl":
					score = z_mse + z_jepa + z_kl
				else:
					score = z_mse + z_jepa

				scores.append(score.cpu())

		scores = torch.cat(scores).numpy()
		self._TCNTransformer__anomaly_score = scores

		if scores.shape[0] < len(data):
			pad_l = int(np.ceil((self.win_size - 1) / 2))
			pad_r = int((self.win_size - 1) // 2)
			scores = np.array([scores[0]] * pad_l + list(scores) + [scores[-1]] * pad_r)
			self._TCNTransformer__anomaly_score = scores

		return scores


def build_default_variants() -> list[AblationVariant]:
	return [
		AblationVariant("Base", {}, {}),
		# Architecture
		AblationVariant("DModel64", {"d_model": 64}, {}),
		AblationVariant("Heads4", {"num_heads": 4}, {}),
		# Horizon
		AblationVariant("Horizon1", {"forecast_steps": 1}, {}),
		AblationVariant("Horizon25", {"forecast_steps": 25}, {}),
		# Masking
		AblationVariant("MaskRatio0p2", {}, {"mask_ratio": 0.2}),
		AblationVariant("MaskRatio0p8", {}, {"mask_ratio": 0.8}),
		# EMA
		AblationVariant("EMA_m99", {}, {"ema_m": 0.99}),
		AblationVariant("EMA_m0", {}, {"ema_m": 0}),
		AblationVariant("EMA_m999", {}, {"ema_m": 0.999}),
		# Training objective
		AblationVariant("ReconOnly", {}, {"train_objective": "rec"}),
		AblationVariant("JEPAOnly_Q", {}, {"train_objective": "jepa"}),
		# Reviewer-style: what exactly is predicted?
		AblationVariant("PredHiddenState", {}, {"train_objective": "pred_h"}),
		# These objectives add extra compute/memory; use a smaller default batch.
		AblationVariant("PredKeys", {"batch_size": 64}, {"train_objective": "pred_k"}),
		AblationVariant("PredValues", {"batch_size": 64}, {"train_objective": "pred_v"}),
		AblationVariant("PredAttnMap_Q", {"batch_size": 32}, {"train_objective": "pred_attn_q"}),
		AblationVariant("PredAttnMap_QK", {"batch_size": 32}, {"train_objective": "pred_attn_qk"}),
		# Scoring
		AblationVariant("Score_MSE", {}, {"score_mode": "mse"}),
		AblationVariant("Score_MSE_JEPA_KL", {}, {"score_mode": "mse+jepa+kl"}),
	]


def get_base_hp(benchmark: str) -> dict:
	base = dict(Optimal_Multi_algo_HP_dict.get("AxonAD", {}))
	if benchmark.upper() == "U":
		base_u = Optimal_Uni_algo_HP_dict.get("AxonAD")
		if isinstance(base_u, dict) and base_u:
			base.update(base_u)
	return base


def main() -> int:
	parser = argparse.ArgumentParser(description="Run AxonAD ablations and save results CSV")
	parser.add_argument("--benchmark", type=str, default="M", choices=["M", "U"], help="M or U")
	parser.add_argument(
		"--split_mode",
		type=str,
		default="tuning",
		help="For M: eval|tuning|telemetry|all. For U: eval|eval_full|tuning|all",
	)
	parser.add_argument("--seed", type=int, default=2024)
	parser.add_argument("--output", type=str, default=None)
	parser.add_argument("--max_datasets", type=int, default=0, help="0 = all, else limit")
	args = parser.parse_args()

	set_random_seed(args.seed)

	benchmark = args.benchmark.upper()
	split_mode = str(args.split_mode).lower()

	data_roots = ["Datasets/TSB-AD-M"] if benchmark == "M" else ["Datasets/TSB-AD-U"]
	file_list_dir = "Datasets/File_List"

	output_file = args.output
	if output_file is None:
		output_file = f"tsb_ad_{benchmark.lower()}_{split_mode}_axonad_ablations_results.csv"

	allowed = load_official_split_filenames(file_list_dir, benchmark, split_mode)

	all_datasets = list_all_csvs(data_roots)
	print(f"Found {len(all_datasets)} dataset files before split filtering")

	if allowed is not None:
		all_datasets = [p for p in all_datasets if os.path.basename(p) in allowed]
		print(f"{len(all_datasets)} dataset files after applying official '{benchmark}-{split_mode}' split filtering")

	if args.max_datasets and args.max_datasets > 0:
		all_datasets = all_datasets[: int(args.max_datasets)]
		print(f"Limiting to {len(all_datasets)} datasets")

	print(f"Final dataset count: {len(all_datasets)}")

	# Resume support
	if os.path.exists(output_file):
		try:
			if os.path.getsize(output_file) > 0:
				existing_df = pd.read_csv(output_file)
				results = existing_df.to_dict("records") if len(existing_df) > 0 else []
				completed_pairs = set(zip(existing_df.get("AD_Name", []), existing_df.get("dataset", [])))
				print(f"Resuming: Found {len(completed_pairs)} completed (variant, dataset) pairs")
			else:
				results, completed_pairs = [], set()
		except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
			print(f"Warning: Could not read existing results file ({e}). Starting fresh.")
			results, completed_pairs = [], set()
	else:
		results, completed_pairs = [], set()

	base_hp = get_base_hp(benchmark)
	variants = build_default_variants()

	for variant in variants:
		set_random_seed(args.seed)
		ad_name = f"AxonAD__{variant.name}"
		print(f"\n{'=' * 72}\nRunning variant: {ad_name}\n{'=' * 72}")

		variant_results = []

		for data_path in tqdm.tqdm(all_datasets, desc=f"AD={ad_name}"):
			filename = os.path.basename(data_path)

			if (ad_name, filename) in completed_pairs:
				continue

			train_idx = 0
			X_train = None
			X_test = None

			try:
				df = pd.read_csv(data_path).dropna()
				X = df.iloc[:, :-1].values.astype(float)
				y = df.iloc[:, -1].astype(int).to_numpy()

				if X.ndim > 1 and X.shape[1] > 1:
					slidingWindow = find_length_rank(X[:, 0].reshape(-1, 1), rank=1)
				else:
					slidingWindow = find_length_rank(X, rank=1)

				train_idx = parse_train_index(filename)
				if train_idx is None:
					train_idx = int(len(X) * 0.3)

				X_train = X[:train_idx, :]
				X_test = X

				hp = dict(base_hp)
				hp.update(variant.constructor_overrides)

				ctor_kwargs = dict(
					win_size=hp.get("win_size", 100),
					feats=X_test.shape[1],
					d_model=hp.get("d_model", 64),
					num_heads=hp.get("num_heads", 4),
					batch_size=hp.get("batch_size", 128),
					epochs=hp.get("epochs", 50),
					lr=hp.get("lr", 1e-3),
					validation_size=hp.get("validation_size", 0.2),
					kl_tail_k=hp.get("kl_tail_k", 5),
					forecast_steps=hp.get("forecast_steps", 1),
				)

				train_objective = variant.model_overrides.get("train_objective", "balanced")
				score_mode = variant.model_overrides.get("score_mode", "mse+jepa")

				clf = TCNTransformerAblation(
					train_objective=train_objective,
					score_mode=score_mode,
					**ctor_kwargs,
				)

				if "mask_ratio" in variant.model_overrides:
					clf.model.mask_ratio = float(variant.model_overrides["mask_ratio"])
				if "mask_block_frac" in variant.model_overrides:
					clf.model.mask_block_frac = float(variant.model_overrides["mask_block_frac"])
				if "ema_m" in variant.model_overrides:
					clf.model.ema_m = float(variant.model_overrides["ema_m"])

				clf.fit(X_train)
				scores = clf.decision_function(X_test)

				metrics = get_metrics(scores, y, slidingWindow=slidingWindow)

				row = {
					"AD_Name": ad_name,
					"dataset": filename,
					"data_len": len(X),
					"train_split": train_idx,
					"benchmark": benchmark,
					"split_mode": split_mode,
				}
				row.update(metrics)
				variant_results.append(row)

			except KeyboardInterrupt:
				print("Interrupted by user. Saving and exiting...")
				pd.DataFrame(results + variant_results).to_csv(output_file, index=False)
				raise

			except Exception as e:
				print(f"\n[ERROR] {ad_name} failed on {filename}: {str(e)}")
				traceback.print_exc()

				zero_metrics = {
					k: 0.0
					for k in [
						"AUC-PR",
						"AUC-ROC",
						"VUS-PR",
						"VUS-ROC",
						"Standard-F1",
						"PA-F1",
						"Event-based-F1",
						"R-based-F1",
						"Affiliation-F",
					]
				}

				row = {
					"AD_Name": ad_name,
					"dataset": filename,
					"data_len": len(X) if "X" in locals() else 0,
					"train_split": train_idx,
					"benchmark": benchmark,
					"split_mode": split_mode,
				}
				row.update(zero_metrics)
				variant_results.append(row)

			finally:
				if "scores" in locals():
					del scores
				if "clf" in locals():
					del clf
				if "X" in locals():
					del X
				if "y" in locals():
					del y
				if "df" in locals():
					del df
				if X_train is not None:
					del X_train
				if X_test is not None:
					del X_test
				gc.collect()
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
				gc.collect()

		results.extend(variant_results)
		pd.DataFrame(results).to_csv(output_file, index=False)
		print(f"✓ Saved results for {ad_name}")

	print(f"\nAll done. Results written to: {output_file}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
