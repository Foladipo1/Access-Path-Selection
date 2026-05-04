#!/bin/bash
#$ -N cs561_large
#$ -pe omp 4
#$ -l h_rt=12:00:00
#$ -l mem_per_core=16G
#$ -l avx512
#$ -P cs561
#$ -o /projectnb/cs561/students/foladipo/CS561-Access-Path-Selection/results/experiment_large.log
#$ -e /projectnb/cs561/students/foladipo/CS561-Access-Path-Selection/results/experiment_large.err
#$ -V

echo "=========================================="
echo "Job started:  $(date)"
echo "Running on:   $(hostname)"
echo "CPU:          $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "Cores (job):  4 (pe omp 4)"
echo "RAM total:    $(free -h | awk '/^Mem/ {print $2}')"
echo "AVX-512:      $(grep -c avx512f /proc/cpuinfo) logical cores with AVX-512"
echo "=========================================="

module load python3/3.10.12

mkdir -p /projectnb/cs561/students/foladipo/CS561-Access-Path-Selection/results

cd /projectnb/cs561/students/foladipo/CS561-Access-Path-Selection
python3 run_experiments_large.py

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="