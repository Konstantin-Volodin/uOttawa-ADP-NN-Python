#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --ntasks=256
#SBATCH --mem-per-cpu=3gb
#SBATCH --job-name=optimsim
#SBATCH --output=optimsim.out
srun python3 optimSimulation.py