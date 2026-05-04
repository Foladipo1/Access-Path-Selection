# Access Path Selection in Modern Columnar DBMSs



## Overview

This repository contains the experimental code and results for our CS561 final
project investigating access path selection in a modern PAX-based columnar DBMS
(DuckDB). We evaluate four access path mechanisms:

- **Zone Maps** — plain sequential scan, DuckDB default
- **Column Sketches** — AVX-512 accelerated scan (custom DuckDB build)
- **CUBIT** — concurrent updatable bitmap index *(implemented by Daniel Silla)*
- **RABIT** — range-efficient bitmap index *(implemented by Daniel Silla)*

Experiments span three dimensions:
- **Selectivity** — real TPC-H SF=3 data across 5 lineitem columns
- **Cardinality** — synthetic data at 6M and 60M rows
- **Distribution** — synthetic data (uniform, Zipf, clustered) at 6M and 60M rows

For full details on methodology, findings, and analysis see our paper in `paper/`.

---

## Prerequisites

### 1. Build the project
```bash
make release
```
Binary will be at `build/release/duckdb`.

### 2. Python dependencies
```bash
pip install numpy matplotlib
```

### 3. AVX-512 support required
Column Sketches require AVX-512 SIMD instructions. On BU SCC request an
AVX-512 node via SGE (`#$ -l avx512`).

---

## Reproducing Sketch vs Baseline Experiments (BU SCC)

These experiments compare Column Sketches against zone map baseline on
synthetic data, varying selectivity, cardinality, and distribution independently.

### 6M rows
```bash
qsub run_experiments.sh
```
Results saved to `results/selectivity_results.csv`, `results/cardinality_results.csv`,
`results/distribution_results.csv`.

### 60M rows
```bash
qsub run_experiments_large.sh
```
Results saved to the corresponding `*_60m.csv` files.

---

### Selectivity sweep (TPC-H SF=3)
```bash
python3 scripts/benchmark_selectivity.py 3 5 1 results/results_selectivity.csv
```

### Cardinality sweep (synthetic, 6M rows)
```bash
python3 scripts/benchmark_cardinality.py 6000000 5 1 results/results_cardinality.csv
```

### Distribution sweep (synthetic, 6M rows)
```bash
python3 scripts/benchmark_distribution.py 6000000 5 1 results/results_distribution.csv
```

### Scan method selection
The scan method is controlled by the `DUCKDB_SCAN_METHOD` environment variable,
set automatically by the benchmark scripts:

| Value | Description |
|---|---|
| `plain` | Zone map sequential scan (baseline) |
| `sketch` | Column Sketch scan |
| `cubit` | CUBIT bitmap index scan |
| `rabit` | RABIT bitmap index scan |

---

## Experiment Details

All experiments use:
```sql
SELECT count(*) FROM lineitem WHERE l_quantity < <threshold>;
```

| Dimension | Fixed | Varied | Values |
|---|---|---|---|
| Selectivity | cardinality=1000, uniform | threshold | 5%, 10%, 20%, 40%, 60%, 80% |
| Cardinality | selectivity=10%, uniform | NDV | 10, 50, 100, 256, 500, 1K, 10K, 100K |
| Distribution | cardinality=1000, selectivity=10% | distribution | uniform, Zipf, clustered |

Run 1 is always a warmup and excluded from reported results.

---

## Hardware

All experiments were run on the Boston University Shared Computing Cluster (SCC):

- **CPU:** Intel Xeon Gold 6242 @ 2.80GHz
- **RAM:** 250GB
- **SIMD:** AVX-512
- **Threads:** 4
- **Scheduler:** SGE

---

## Key Findings

- Column Sketches provide no measurable advantage over zone maps in any tested configuration
- Data layout is the dominant factor — clustered data is 30× faster for all methods
- CUBIT and RABIT show no consistent advantage; bitmap indexes hurt on clustered data
- The theoretical sketch degradation above NDV=256 does not materialize under PAX-based vectorized execution
