#!/usr/bin/env python3
"""
Aggregate benchmark results: calculate mean performance per model across all datasets.
"""

import pandas as pd
import numpy as np
import os
import glob

def aggregate_results(input_file, output_file=None):
    """
    Calculate mean performance for each model across all datasets.
    
    Args:
        input_file: Path to the results CSV file
        output_file: Path to save aggregated results (default: add '_aggregated' to input filename)
    """
    # Read results
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} results from {input_file}")
    print(f"Models: {df['AD_Name'].nunique()}")
    print(f"Datasets: {df['dataset'].nunique()}")
    
    # Identify metric columns (exclude metadata columns)
    metadata_cols = ['AD_Name', 'dataset', 'data_len', 'train_split', 'benchmark', 'split_mode']
    metric_cols = [col for col in df.columns if col not in metadata_cols]
    
    print(f"\nMetrics found: {metric_cols}")
    
    # Calculate mean performance per model
    aggregated = df.groupby('AD_Name')[metric_cols].agg(['mean', 'std', 'count']).reset_index()
    
    # Flatten column names
    aggregated.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                          for col in aggregated.columns.values]
    
    # Rename for clarity
    aggregated = aggregated.rename(columns={'AD_Name': 'Model'})
    
    # Sort by a primary metric (e.g., VUS-PR mean)
    if 'VUS-PR_mean' in aggregated.columns:
        aggregated = aggregated.sort_values('VUS-PR_mean', ascending=False)
    elif 'AUC-PR_mean' in aggregated.columns:
        aggregated = aggregated.sort_values('AUC-PR_mean', ascending=False)
    
    # Set output filename
    if output_file is None:
        base = os.path.splitext(input_file)[0]
        output_file = f"{base}_aggregated.csv"
    
    # Save
    aggregated.to_csv(output_file, index=False)
    print(f"\n✓ Saved aggregated results to {output_file}")
    
    # Display summary
    print("\n" + "="*80)
    print("Top 10 Models by VUS-PR (mean):")
    print("="*80)
    
    # Show compact summary
    display_cols = ['Model'] + [col for col in aggregated.columns if '_mean' in col]
    summary = aggregated[display_cols].head(10)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(summary.to_string(index=False))
    
    return aggregated

if __name__ == '__main__':
    # Find the most recent results file
    result_files = glob.glob('tsb_ad_*_results.csv')
    
    if not result_files:
        print("Error: No results files found (tsb_ad_*_results.csv)")
        exit(1)
    
    # Use the most recently modified file
    input_file = max(result_files, key=os.path.getmtime)
    print(f"Using results file: {input_file}\n")
    
    aggregate_results(input_file)
