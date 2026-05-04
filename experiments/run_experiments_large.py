#!/usr/bin/env python3
"""
CS561 Access Path Selection Experiments - Large Scale (60M rows)
Compares Column Sketches vs Baseline Scan (Zone Maps) across:
  1. Selectivity   (l_quantity, cardinality=1000, uniform, vary threshold)
  2. Cardinality   (uniform distribution, fixed 10% selectivity)
  3. Distribution  (fixed cardinality=1000, fixed 10% selectivity)

Scale: 60M rows (vs 6M in run_experiments.py) to stress memory bandwidth
Hardware: Intel Xeon Gold 6242 @ 2.80GHz, 16 cores, 250GB RAM, AVX-512
Threads:  4 (PRAGMA threads=4)
Runs:     3 (averaged, after 1 warmup run)
"""

import subprocess
import os
import re
import csv
import shutil
import tempfile
import numpy as np

DUCKDB_BIN  = "/projectnb/cs561/students/foladipo/CS561-Access-Path-Selection/build/release/duckdb"
RESULTS_DIR = "/projectnb/cs561/students/foladipo/CS561-Access-Path-Selection/results"

N_ROWS   = 60_000_000
N_RUNS   = 3
THREADS  = 4

SELECTIVITY_LEVELS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
CARDINALITY_LEVELS = [10, 50, 100, 256, 500, 1000, 10000, 100000]
DISTRIBUTIONS      = ['uniform', 'zipf', 'clustered']

FIXED_SELECTIVITY  = 0.10
FIXED_CARDINALITY  = 1000

LINEITEM_SCHEMA = """(
    l_orderkey      INTEGER,
    l_partkey       INTEGER,
    l_suppkey       INTEGER,
    l_linenumber    INTEGER,
    l_quantity      INTEGER,
    l_extendedprice DOUBLE,
    l_discount      DOUBLE,
    l_tax           DOUBLE,
    l_returnflag    VARCHAR,
    l_linestatus    VARCHAR,
    l_shipdate      DATE,
    l_commitdate    DATE,
    l_receiptdate   DATE,
    l_shipinstruct  VARCHAR,
    l_shipmode      VARCHAR,
    l_comment       VARCHAR
)"""

RETURNFLAG   = ['N', 'A', 'R']
LINESTATUS   = ['O', 'F']
SHIPINSTRUCT = ['DELIVER IN PERSON', 'COLLECT COD', 'NONE', 'TAKE BACK RETURN']
SHIPMODE     = ['REG AIR', 'AIR', 'RAIL', 'SHIP', 'TRUCK', 'MAIL', 'FOB']

def generate_qty(distribution, cardinality, n_rows):
    if distribution == 'uniform':
        return np.random.randint(0, cardinality, n_rows, dtype=np.int32)
    elif distribution == 'zipf':
        raw = np.random.zipf(1.5, n_rows).astype(np.int64)
        return np.clip(raw - 1, 0, cardinality - 1).astype(np.int32)
    elif distribution == 'clustered':
        return np.sort(np.random.randint(0, cardinality, n_rows, dtype=np.int32))
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

def get_threshold(qty_values, selectivity):
    sorted_vals = np.sort(qty_values)
    idx = int(selectivity * len(sorted_vals))
    threshold = int(sorted_vals[min(idx, len(sorted_vals) - 1)])
    if threshold == 0:
        threshold = int(sorted_vals[min(idx * 2, len(sorted_vals) - 1)]) + 1
    return threshold

def write_csv(qty_values, filepath):
    import pandas as pd
    n = len(qty_values)
    rng = np.random.default_rng(42)

    base_date  = np.datetime64('1992-01-01')
    date_range = (np.datetime64('1998-12-31') - base_date).astype(int)
    ship_offsets = rng.integers(0, date_range - 90, n)
    shipdates    = [str(base_date + np.timedelta64(int(d), 'D')) for d in ship_offsets]
    commitdates  = [str(base_date + np.timedelta64(int(d) + int(rng.integers(30, 90)), 'D')) for d in ship_offsets]
    receiptdates = [str(base_date + np.timedelta64(int(d) + int(rng.integers(1, 30)), 'D')) for d in ship_offsets]

    df = pd.DataFrame({
        'l_orderkey':      rng.integers(1, 60_000_000, n).astype(np.int32),
        'l_partkey':       rng.integers(1, 200_000, n).astype(np.int32),
        'l_suppkey':       rng.integers(1, 10_000, n).astype(np.int32),
        'l_linenumber':    rng.integers(1, 7, n).astype(np.int32),
        'l_quantity':      qty_values.astype(np.int32),
        'l_extendedprice': np.round(rng.uniform(900.0, 104_000.0, n), 2),
        'l_discount':      np.round(rng.uniform(0.0, 0.10, n), 2),
        'l_tax':           np.round(rng.uniform(0.0, 0.08, n), 2),
        'l_returnflag':    rng.choice(RETURNFLAG, n),
        'l_linestatus':    rng.choice(LINESTATUS, n),
        'l_shipdate':      shipdates,
        'l_commitdate':    commitdates,
        'l_receiptdate':   receiptdates,
        'l_shipinstruct':  rng.choice(SHIPINSTRUCT, n),
        'l_shipmode':      rng.choice(SHIPMODE, n),
        'l_comment':       ['comment'] * n,
    })
    df.to_csv(filepath, index=False)

def run_duckdb(db_path, sql, timeout=3600):
    cmd = [DUCKDB_BIN, db_path]
    result = subprocess.run(cmd, input=sql, capture_output=True, text=True, timeout=timeout)
    return result.stdout, result.stderr

def load_and_query(db_path, table_name, csv_path, threshold):
    """Load table, run 1 warmup query, then N_RUNS timed queries in a single session."""
    query = f"SELECT count(*) FROM {table_name} WHERE l_quantity < {threshold};\n"
    sql = (
        f"PRAGMA threads={THREADS};\n"
        f"DROP TABLE IF EXISTS {table_name};\n"
        f"CREATE TABLE {table_name} {LINEITEM_SCHEMA};\n"
        f"COPY {table_name} FROM '{csv_path}' (HEADER TRUE);\n"
        + query +
        f".timer on\n"
        + query * N_RUNS
    )
    stdout, stderr = run_duckdb(db_path, sql)
    times = [float(m) * 1000.0
             for m in re.findall(r'Run Time \(s\): real (\d+\.?\d*)', stdout)]
    if len(times) != N_RUNS:
        print(f"    [WARN] Expected {N_RUNS} timings, got {len(times)}.")
        if stderr.strip():
            print(f"    stderr: {stderr.strip()[:300]}")
        if stdout.strip():
            print(f"    stdout: {stdout.strip()[:500]}")
    return times

def run_experiment(qty_values, threshold, label, tmp_dir):
    csv_path    = os.path.join(tmp_dir, f"{label}.csv")
    sketch_db   = os.path.join(tmp_dir, f"{label}_sketch.duckdb")
    baseline_db = os.path.join(tmp_dir, f"{label}_baseline.duckdb")

    print(f"    Writing CSV ({N_ROWS:,} rows)...")
    write_csv(qty_values, csv_path)

    print(f"    Running sketched session (lineitem)...")
    sketch_times = load_and_query(sketch_db, "lineitem", csv_path, threshold)

    print(f"    Running baseline session (lineitem_base)...")
    baseline_times = load_and_query(baseline_db, "lineitem_base", csv_path, threshold)

    for f in [csv_path, sketch_db, baseline_db]:
        if os.path.exists(f):
            os.remove(f)

    return sketch_times, baseline_times

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="cs561_large_exp_", dir=RESULTS_DIR)

    print("=" * 60)
    print("CS561 Access Path Selection Experiments - 60M rows")
    print(f"DuckDB: {DUCKDB_BIN}")
    print(f"Rows: {N_ROWS:,}  |  Runs: {N_RUNS} + 1 warmup  |  Threads: {THREADS}")
    print("=" * 60)

    # 1. Selectivity
    print("\n>>> SELECTIVITY EXPERIMENT (cardinality=1000, uniform)")
    sel_rows = []
    qty_sel = generate_qty('uniform', FIXED_CARDINALITY, N_ROWS)
    for sel in SELECTIVITY_LEVELS:
        threshold  = get_threshold(qty_sel, sel)
        actual_sel = float(np.sum(qty_sel < threshold)) / N_ROWS
        print(f"\n  Selectivity = {sel*100:.0f}%  |  Threshold: {threshold}  |  Actual: {actual_sel*100:.1f}%")
        sketch_t, baseline_t = run_experiment(qty_sel, threshold, f"sel_{int(sel*100)}", tmp_dir)
        if sketch_t and baseline_t:
            print(f"  Sketch avg:   {np.mean(sketch_t):.1f} ms")
            print(f"  Baseline avg: {np.mean(baseline_t):.1f} ms")
        for i, t in enumerate(sketch_t):
            sel_rows.append([sel, threshold, actual_sel, 'sketch',   i+1, round(t, 2)])
        for i, t in enumerate(baseline_t):
            sel_rows.append([sel, threshold, actual_sel, 'baseline', i+1, round(t, 2)])
    sel_csv = os.path.join(RESULTS_DIR, 'selectivity_results_60m.csv')
    with open(sel_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['selectivity', 'threshold', 'actual_selectivity', 'method', 'run', 'time_ms'])
        w.writerows(sel_rows)
    print(f"\n  Saved: {sel_csv}")

    # 2. Cardinality
    print("\n>>> CARDINALITY EXPERIMENT (uniform, 10% selectivity)")
    card_rows = []
    for cardinality in CARDINALITY_LEVELS:
        print(f"\n  Cardinality = {cardinality:,}")
        qty       = generate_qty('uniform', cardinality, N_ROWS)
        threshold = get_threshold(qty, FIXED_SELECTIVITY)
        actual_sel = float(np.sum(qty < threshold)) / N_ROWS
        print(f"  Threshold: {threshold}  |  Actual selectivity: {actual_sel*100:.1f}%")
        sketch_t, baseline_t = run_experiment(qty, threshold, f"card_{cardinality}", tmp_dir)
        if sketch_t and baseline_t:
            print(f"  Sketch avg:   {np.mean(sketch_t):.1f} ms")
            print(f"  Baseline avg: {np.mean(baseline_t):.1f} ms")
        for i, t in enumerate(sketch_t):
            card_rows.append([cardinality, threshold, actual_sel, 'sketch',   i+1, round(t, 2)])
        for i, t in enumerate(baseline_t):
            card_rows.append([cardinality, threshold, actual_sel, 'baseline', i+1, round(t, 2)])
    card_csv = os.path.join(RESULTS_DIR, 'cardinality_results_60m.csv')
    with open(card_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cardinality', 'threshold', 'actual_selectivity', 'method', 'run', 'time_ms'])
        w.writerows(card_rows)
    print(f"\n  Saved: {card_csv}")

    # 3. Distribution
    print("\n>>> DISTRIBUTION EXPERIMENT (cardinality=1000, 10% selectivity)")
    dist_rows = []
    for dist in DISTRIBUTIONS:
        print(f"\n  Distribution = {dist}")
        qty       = generate_qty(dist, FIXED_CARDINALITY, N_ROWS)
        threshold = get_threshold(qty, FIXED_SELECTIVITY)
        actual_sel = float(np.sum(qty < threshold)) / N_ROWS
        print(f"  Threshold: {threshold}  |  Actual selectivity: {actual_sel*100:.1f}%")
        sketch_t, baseline_t = run_experiment(qty, threshold, f"dist_{dist}", tmp_dir)
        if sketch_t and baseline_t:
            print(f"  Sketch avg:   {np.mean(sketch_t):.1f} ms")
            print(f"  Baseline avg: {np.mean(baseline_t):.1f} ms")
        for i, t in enumerate(sketch_t):
            dist_rows.append([dist, threshold, actual_sel, 'sketch',   i+1, round(t, 2)])
        for i, t in enumerate(baseline_t):
            dist_rows.append([dist, threshold, actual_sel, 'baseline', i+1, round(t, 2)])
    dist_csv = os.path.join(RESULTS_DIR, 'distribution_results_60m.csv')
    with open(dist_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['distribution', 'threshold', 'actual_selectivity', 'method', 'run', 'time_ms'])
        w.writerows(dist_rows)
    print(f"\n  Saved: {dist_csv}")

    shutil.rmtree(tmp_dir)
    print("\n>>> ALL EXPERIMENTS COMPLETE")

if __name__ == "__main__":
    main()