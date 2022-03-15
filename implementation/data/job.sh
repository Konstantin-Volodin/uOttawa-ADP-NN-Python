#!/bin/bash
#SBATCH --time=01-00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --mem=0
#SBATCH --job-name=optimSimulation
#SBATCH --output=optimSimulation.out
python3 optimSimulation.py
