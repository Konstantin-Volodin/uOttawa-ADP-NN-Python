#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=256
#SBATCH --mem=3gb
#SBATCH --job-name=optimsim
#SBATCH --output=optimsim.out
srun optimSimulation.py