#!/usr/bin/env python3
"""
CS561 Access Path Selection Experiments
Compares Column Sketches vs Baseline Scan (Zone Maps) across:
  1. Cardinality levels (uniform distribution, fixed 10% selectivity)
  2. Data distributions (fixed cardinality=1000, fixed 10% selectivity)

Hardware: Intel Xeon Gold 6242 @ 2.80GHz, 16 cores, 250GB RAM, AVX-512
Threads:  4 (PRAGMA threads=4)
Rows:     6,000,000
Runs:     3 (averaged)
"""

import subprocess
import os
import re
import csv
import shutil
import tempfile
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

DUCKDB_BIN = "/projectnb/cs561/students/foladipo/CS561-Access-Path-Selection/build/release/duckdb"
RESULTS_DIR = "/projectnb/cs561/students/foladipo/CS561-Access-Path-Selection/results"

N_ROWS      = 6_000_000
N_RUNS      = 3
THREADS     = 4
SELECTIVITY = 0.10  # 10%

CARDINALITY_LEVELS = [10, 50, 100, 256, 500, 1000, 10000, 100000]
DISTRIBUTIONS      = ['uniform', 'zipf', 'clustered']
DIST_CARDINALITY   = 1000   # fixed cardinality for distribution experiment

# Full lineitem schema — required so sketch activates on l_quantity (index 4)
LINEITEM_SCHEMA = """(
    l_orderkey    INTEGER,
    l_partkey     INTEGER,
    l_suppkey     INTEGER,
    l_linenumber  INTEGER,
    l_quantity    INTEGER,
    l_extendedprice DOUBLE,
    l_discount    DOUBLE,
    l_tax         DOUBLE,
    l_returnflag  VARCHAR,
    l_linestatus  VARCHAR,
    l_shipdate    DATE,
    l_commitdate  DATE,
    l_receiptdate DATE,
    l_shipinstruct VARCHAR,
    l_shipmode    VARCHAR,
    l_comment     VARCHAR
)"""

# ── Data Generation ───────────────────────────────────────────────────────────

def generate_qty(distribution, cardinality, n_rows):
    """Generate l_quantity values for a given distribution and cardinality."""
    if distribution == 'uniform':
        return np.random.randint(0, cardinality, n_rows, dtype=np.int32)

    elif distribution == 'zipf':
        # Zipf(a=1.5): heavy skew toward low values, realistic for real data
        raw = np.random.zipf(1.5, n_rows).astype(np.int64)
        # Map to [0, cardinality-1]
        raw = np.clip(raw - 1, 0, cardinality - 1).astype(np.int32)
        return raw

    elif distribution == 'clustered':
        # Sorted values: best case for zone map pruning (tight per-row-group ranges)
        return np.sort(np.random.randint(0, cardinality, n_rows, dtype=np.int32))

    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def get_threshold(qty_values):
    """Return threshold value achieving ~SELECTIVITY fraction of rows."""
    return int(np.percentile(qty_values, SELECTIVITY * 100))


def write_csv(qty_values, filepath):
    """Write full 16-column lineitem CSV — required for COPY to trigger sketchAppend."""
    import pandas as pd
    n = len(qty_values)
    df = pd.DataFrame({
        'l_orderkey':     np.arange(n, dtype=np.int32),
        'l_partkey':      np.ones(n, dtype=np.int32),
        'l_suppkey':      np.ones(n, dtype=np.int32),
        'l_linenumber':   np.ones(n, dtype=np.int32),
        'l_quantity':     qty_values.astype(np.int32),
        'l_extendedprice': np.ones(n, dtype=np.float32),
        'l_discount':     np.zeros(n, dtype=np.float32),
        'l_tax':          np.zeros(n, dtype=np.float32),
        'l_returnflag':   'N',
        'l_linestatus':   'O',
        'l_shipdate':     '1994-01-01',
        'l_commitdate':   '1994-02-01',
        'l_receiptdate':  '1994-03-01',
        'l_shipinstruct': 'A',
        'l_shipmode':     'A',
        'l_comment':      'A',
    })
    df.to_csv(filepath, index=False)


# ── DuckDB Execution ──────────────────────────────────────────────────────────

def run_duckdb(db_path, sql, timeout=600):
    """Pipe SQL to custom DuckDB CLI binary, return (stdout, stderr)."""
    cmd = [DUCKDB_BIN, db_path]
    result = subprocess.run(
        cmd,
        input=sql,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result.stdout, result.stderr


def load_table(db_path, table_name, csv_path):
    """Create table and COPY data in. For 'lineitem', this triggers sketchAppend."""
    sql = f"""
PRAGMA threads={THREADS};
DROP TABLE IF EXISTS {table_name};
CREATE TABLE {table_name} {LINEITEM_SCHEMA};
COPY {table_name} FROM '{csv_path}' (HEADER TRUE);
"""
    stdout, stderr = run_duckdb(db_path, sql)
    if stderr.strip():
        print(f"    [WARN] Load stderr: {stderr.strip()[:300]}")


def timed_query(db_path, table_name, threshold):
    """Run timed SELECT query, return execution time in ms or None on failure."""
    sql = f"""PRAGMA threads={THREADS};
.timer on
SELECT count(*) FROM {table_name} WHERE l_quantity < {threshold};
"""
    stdout, stderr = run_duckdb(db_path, sql)
    match = re.search(r'Run Time \(s\): real (\d+\.?\d*)', stdout)
    if match:
        return float(match.group(1)) * 1000.0  # convert to ms
    print(f"    [WARN] Could not parse timing. stdout: {stdout[:200]}")
    return None


# ── Experiment Runner ─────────────────────────────────────────────────────────

def run_experiment(qty_values, threshold, label, tmp_dir):
    """
    For a given qty array:
      - Load into 'lineitem' (sketched) and 'lineitem_base' (baseline)
      - Run N_RUNS timed queries on each
    Returns (sketch_times_ms, baseline_times_ms)
    """
    csv_path     = os.path.join(tmp_dir, f"{label}.csv")
    sketch_db    = os.path.join(tmp_dir, f"{label}_sketch.duckdb")
    baseline_db  = os.path.join(tmp_dir, f"{label}_baseline.duckdb")

    print(f"    Writing CSV ({N_ROWS:,} rows)...")
    write_csv(qty_values, csv_path)

    print(f"    Loading sketched table (lineitem)...")
    load_table(sketch_db, "lineitem", csv_path)

    print(f"    Loading baseline table (lineitem_base)...")
    load_table(baseline_db, "lineitem_base", csv_path)

    # CSV no longer needed
    os.remove(csv_path)

    sketch_times   = []
    baseline_times = []

    for run in range(1, N_RUNS + 1):
        print(f"    Run {run}/{N_RUNS}...")
        t = timed_query(sketch_db,   "lineitem",      threshold)
        if t is not None:
            sketch_times.append(t)
        t = timed_query(baseline_db, "lineitem_base", threshold)
        if t is not None:
            baseline_times.append(t)

    # Cleanup db files
    for f in [sketch_db, baseline_db]:
        if os.path.exists(f):
            os.remove(f)

    return sketch_times, baseline_times


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="cs561_exp_")

    print("=" * 60)
    print("CS561 Access Path Selection Experiments")
    print(f"DuckDB binary: {DUCKDB_BIN}")
    print(f"Results dir:   {RESULTS_DIR}")
    print(f"Rows: {N_ROWS:,}  |  Runs: {N_RUNS}  |  Threads: {THREADS}")
    print(f"Selectivity: {SELECTIVITY*100:.0f}%")
    print("=" * 60)

    # ── 1. Cardinality Experiment ─────────────────────────────────────────────
    print("\n>>> CARDINALITY EXPERIMENT (uniform distribution)")
    card_rows = []

    for cardinality in CARDINALITY_LEVELS:
        print(f"\n  Cardinality = {cardinality:,}")
        qty       = generate_qty('uniform', cardinality, N_ROWS)
        threshold = get_threshold(qty)
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

    card_csv = os.path.join(RESULTS_DIR, 'cardinality_results.csv')
    with open(card_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cardinality', 'threshold', 'actual_selectivity', 'method', 'run', 'time_ms'])
        w.writerows(card_rows)
    print(f"\n  Saved: {card_csv}")

    # ── 2. Distribution Experiment ────────────────────────────────────────────
    print("\n>>> DISTRIBUTION EXPERIMENT (cardinality=1000)")
    dist_rows = []

    for dist in DISTRIBUTIONS:
        print(f"\n  Distribution = {dist}")
        qty       = generate_qty(dist, DIST_CARDINALITY, N_ROWS)
        threshold = get_threshold(qty)
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

    dist_csv = os.path.join(RESULTS_DIR, 'distribution_results.csv')
    with open(dist_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['distribution', 'threshold', 'actual_selectivity', 'method', 'run', 'time_ms'])
        w.writerows(dist_rows)
    print(f"\n  Saved: {dist_csv}")

    shutil.rmtree(tmp_dir)
    print("\n>>> ALL EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
