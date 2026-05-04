#!/usr/bin/env python3
"""
CS561 Access Path Selection - TPC-H SF=3 Selectivity Experiment
Compares Column Sketches vs Baseline Scan (Zone Maps)
on real TPC-H lineitem data (SF=3, ~18M rows)

Single session per method: load once, run all thresholds in same session.
Threads: 1 (matching teammate's setup)
Runs: 5 per threshold, median reported
"""

import subprocess
import os
import re
import csv
import tempfile
import numpy as np

DUCKDB_BIN  = "/projectnb/cs561/students/foladipo/CS561-Access-Path-Selection/build/release/duckdb"
RESULTS_DIR = "/projectnb/cs561/students/foladipo/CS561-Access-Path-Selection/results"
CSV_PATH    = "/projectnb/cs561/students/foladipo/CS561-Access-Path-Selection/results/lineitem_sf3.csv"

N_RUNS  = 5
THREADS = 1

# l_quantity range 1-50, ~uniform
# Each threshold T gives selectivity (T-1)/50
THRESHOLDS = [
    (4,  "~5%"),
    (6,  "~10%"),
    (11, "~20%"),
    (21, "~40%"),
    (31, "~60%"),
    (41, "~80%"),
]

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

def run_duckdb(db_path, sql, timeout=7200):
    cmd = [DUCKDB_BIN, db_path]
    result = subprocess.run(cmd, input=sql, capture_output=True, text=True, timeout=timeout)
    return result.stdout, result.stderr

def run_all_thresholds(table_name, db_path):
    """
    Single session: load CSV, warmup on first threshold,
    then run all thresholds N_RUNS times each with .timer on.
    Returns dict: threshold -> list of N_RUNS times in ms
    """
    # Build SQL: load, warmup, then timer on, then all queries
    warmup_query = f"SELECT * FROM {table_name} WHERE l_quantity < {THRESHOLDS[0][0]};\n"

    sql = (
        f"PRAGMA threads={THREADS};\n"
        f"DROP TABLE IF EXISTS {table_name};\n"
        f"CREATE TABLE {table_name} {LINEITEM_SCHEMA};\n"
        f"COPY {table_name} FROM '{CSV_PATH}' "
        f"(HEADER TRUE, TYPES {{'l_quantity': 'INTEGER'}});\n"
        + warmup_query  # untimed warmup
        + ".timer on\n"
    )

    # For each threshold, run N_RUNS times
    for threshold, _ in THRESHOLDS:
        query = f"SELECT * FROM {table_name} WHERE l_quantity < {threshold};\n"
        sql += query * N_RUNS

    print(f"  Running {table_name} session (load + all thresholds)...")
    stdout, stderr = run_duckdb(db_path, sql)

    if stderr.strip():
        print(f"  [WARN] stderr: {stderr.strip()[:300]}")

    # Parse all timings in order
    all_times = [float(m) * 1000.0
                 for m in re.findall(r'Run Time \(s\): real (\d+\.?\d*)', stdout)]

    expected = len(THRESHOLDS) * N_RUNS
    if len(all_times) != expected:
        print(f"  [WARN] Expected {expected} timings, got {len(all_times)}")
        if stdout.strip():
            print(f"  stdout snippet: {stdout[:500]}")

    # Split timings by threshold
    results = {}
    for i, (threshold, label) in enumerate(THRESHOLDS):
        start = i * N_RUNS
        results[threshold] = all_times[start:start + N_RUNS]

    return results

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="cs561_tpch_", dir=RESULTS_DIR)

    sketch_db   = os.path.join(tmp_dir, "sketch.duckdb")
    baseline_db = os.path.join(tmp_dir, "baseline.duckdb")

    print("=" * 60)
    print("CS561 TPC-H SF=3 Selectivity Experiment")
    print(f"Data: {CSV_PATH}")
    print(f"Runs: {N_RUNS} per threshold  |  Threads: {THREADS}")
    print("=" * 60)

    sketch_results   = run_all_thresholds("lineitem",      sketch_db)
    baseline_results = run_all_thresholds("lineitem_base", baseline_db)

    # Print summary
    print("\n  Results summary:")
    for threshold, label in THRESHOLDS:
        st = sketch_results.get(threshold, [])
        bt = baseline_results.get(threshold, [])
        if st and bt:
            print(f"  {label}: Sketch median={np.median(st):.1f}ms  Baseline median={np.median(bt):.1f}ms")

    # Save CSV
    rows = []
    for threshold, label in THRESHOLDS:
        for i, t in enumerate(sketch_results.get(threshold, [])):
            rows.append([threshold, label, 'sketch',   i+1, round(t, 2)])
        for i, t in enumerate(baseline_results.get(threshold, [])):
            rows.append([threshold, label, 'baseline', i+1, round(t, 2)])

    out_csv = os.path.join(RESULTS_DIR, 'selectivity_results_tpch.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['threshold', 'selectivity_label', 'method', 'run', 'time_ms'])
        w.writerows(rows)
    print(f"\n  Saved: {out_csv}")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir)
    print("\n>>> EXPERIMENT COMPLETE")

if __name__ == "__main__":
    main()