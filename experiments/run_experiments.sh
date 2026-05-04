#!/bin/bash
#$ -N cs561_experiments
#$ -pe omp 4
#$ -l h_rt=04:00:00
#$ -l mem_per_core=8G
#$ -o /projectnb/cs561/students/foladipo/CS561-Access-Path-Selection/results/experiment.log
#$ -e /projectnb/cs561/students/foladipo/CS561-Access-Path-Selection/results/experiment.err
#$ -V

echo "=========================================="
echo "Job started:  $(date)"
echo "Running on:   $(hostname)"
echo "CPU:          $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "Cores (job):  4 (pe omp 4)"
echo "RAM total:    $(free -h | awk '/^Mem/ {print $2}')"
echo "AVX-512:      $(grep -c avx512f /proc/cpuinfo) logical cores with AVX-512"
echo "=========================================="

# Load python module (BU SCC)
module load python3/3.10.12

# Create results dir if it doesn't exist
mkdir -p /projectnb/cs561/students/foladipo/CS561-Access-Path-Selection/results

# Run experiments
cd /projectnb/cs561/students/foladipo/CS561-Access-Path-Selection
python3 run_experiments.py

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
