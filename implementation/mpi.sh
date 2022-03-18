#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --cpus-per-task=128
#SBATCH --nodes=2
#SBATCH --mem=0
#SBATCH --job-name=optimsim
#SBATCH --output=optim-optimsim.out
mpirun -n 128 python3 mpi.py