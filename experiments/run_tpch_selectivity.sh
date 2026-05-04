
#!/bin/bash

#$ -N cs561_tpch

#$ -pe omp 4

#$ -l h_rt=06:00:00

#$ -l mem_per_core=16G

#$ -l avx512

#$ -P cs561

#$ -o /projectnb/cs561/students/foladipo/CS561-Access-Path-Selection/results/experiment_tpch.log

#$ -e /projectnb/cs561/students/foladipo/CS561-Access-Path-Selection/results/experiment_tpch.err

#$ -V



echo "=========================================="

echo "Job started:  $(date)"

echo "Running on:   $(hostname)"

echo "CPU:          $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"

echo "AVX-512:      $(grep -c avx512f /proc/cpuinfo) logical cores with AVX-512"

echo "=========================================="



module load python3/3.10.12



cd /projectnb/cs561/students/foladipo/CS561-Access-Path-Selection

python3 run_tpch_selectivity.py



echo "=========================================="

echo "Job finished: $(date)"

echo "=========================================="

