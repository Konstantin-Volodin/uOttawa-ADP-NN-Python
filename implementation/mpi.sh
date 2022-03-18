#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=3G
#SBATCH --job-name=testmpi
#SBATCH --output=mpi.out
load module python/3.9.6 gurobi/9.5.0 mpi4py
source /home/kvolodin/python-venv/bin/activate
srun python3 mpi.py