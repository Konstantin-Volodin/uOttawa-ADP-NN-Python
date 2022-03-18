#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --ntasks=128
#SBATCH --mem-per-cpu=3G
#SBATCH --job-name=testmpi
#SBATCH --output=mpi.out
srun python3 mpi.py