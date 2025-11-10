""" kmeans.py

Updated script to run different k-means variants and evaluate using Normalized Mutual
Information (NMI). Designed to be run as a single `.py` file (no notebooks required).

Usage (example):
    python3 kmeans_simple.py --k-min 2 --k-max 6 --repeats 5

Requirements:
    pip install scikit-learn pandas numpy

Notes for the lab:
 - The mock lab may not provide sudo; create a virtualenv if needed.
 - This script will print a CSV `kmeans_results.csv` with aggregated NMI scores.
"""

import argparse
import csv
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    print("Missing scikit-learn or related packages. Please install with:\n    pip install scikit-learn pandas numpy")
    raise


def load_iris(path):
    df = pd.read_csv(path)
    # Determine label column heuristically
    if 'Actual Cluster Label' in df.columns:
        label_col = 'Actual Cluster Label'
    elif 'Species' in df.columns:
        label_col = 'Species'
    else:
        # assume last column is label
        label_col = df.columns[-1]

    # If there's an Id column, drop it
    X_df = df.drop([c for c in ['Id', label_col] if c in df.columns], axis=1)
    X = X_df.values
    y, uniques = pd.factorize(df[label_col])
    return X, y, X_df.columns.tolist()


def run_experiments(X, y, k_min=2, k_max=6, repeats=5, algorithms=None, inits=None, out_csv='kmeans_results.csv'):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if algorithms is None:
        algorithms = {
            'KMeans': KMeans,
            'MiniBatchKMeans': MiniBatchKMeans,
        }

    if inits is None:
        inits = ['k-means++', 'random']

    results = []

    total_runs = len(range(k_min, k_max + 1)) * len(algorithms) * len(inits) * repeats
    run_idx = 0
    start_time = time.time()

    for k in range(k_min, k_max + 1):
        for alg_name, alg_cls in algorithms.items():
            for init in inits:
                scores = []
                for r in range(repeats):
                    run_idx += 1
                    seed = None if r is None else (r + 1) * 42
                    # instantiate
                    params = dict(n_clusters=k, init=init, random_state=seed)
                    # n_init explicit for compatibility and stability
                    if alg_name == 'KMeans':
                        params.setdefault('n_init', 10)
                    else:
                        # MiniBatchKMeans also accepts n_init
                        params.setdefault('n_init', 10)

                    model = alg_cls(**params)
                    labels = model.fit_predict(Xs)
                    nmi = normalized_mutual_info_score(y, labels)
                    scores.append(nmi)

                    elapsed = time.time() - start_time
                    print(f"Run {run_idx}/{total_runs}: k={k} alg={alg_name} init={init} r={r+1} NMI={nmi:.4f} (elapsed {elapsed:.1f}s)")

                scores = np.array(scores)
                results.append({
                    'k': k,
                    'algorithm': alg_name,
                    'init': init,
                    'repeats': repeats,
                    'nmi_mean': float(scores.mean()),
                    'nmi_std': float(scores.std()),
                })

    # Save results to CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['k', 'algorithm', 'init', 'repeats', 'nmi_mean', 'nmi_std'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    return results


def print_results_table(results):
    df = pd.DataFrame(results)
    print('\nAggregated results:')
    print(df.sort_values(['k', 'algorithm', 'init']).to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(description='Run KMeans variants and report NMI on iris dataset')
    p.add_argument('--data', default='./iris.csv', help='Path to iris CSV')
    p.add_argument('--k-min', type=int, default=2)
    p.add_argument('--k-max', type=int, default=6)
    p.add_argument('--repeats', type=int, default=5, help='Number of random restarts to average over')
    p.add_argument('--out', default='kmeans_results.csv', help='Output CSV file for aggregated results')
    return p.parse_args()


def main():
    args = parse_args()

    try:
        X, y, feature_names = load_iris(args.data)
    except FileNotFoundError:
        print(f"Data file not found: {args.data}")
        sys.exit(2)

    results = run_experiments(X, y, k_min=args.k_min, k_max=args.k_max, repeats=args.repeats, out_csv=args.out)
    print_results_table(results)
    print(f"\nWrote aggregated results to {args.out}")


if __name__ == '__main__':
    main()
