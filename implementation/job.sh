#!/bin/bash
#SBATCH --time=02-00
#SBATCH --cpus-per-task=44
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --job-name=optimsim
#SBATCH --output=optim-optimsim.out
python3 optimSimulation.py