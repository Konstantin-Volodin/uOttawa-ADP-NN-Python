#!/bin/bash
#SBATCH --time=02-00
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --mem=0
python3 optimSimulation.py