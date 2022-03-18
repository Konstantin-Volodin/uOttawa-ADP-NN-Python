#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --ntasks=128
#SBATCH --nodes=2
#SBATCH --mem=0
#SBATCH --job-name=testmpi
#SBATCH --output=optim-mpi.out
mpirun -n 128 python3 mpi.py