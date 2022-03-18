#!/bin/bash
#SBATCH --time=00-05
#SBATCH --nodes=10
#SBATCH --ntasks=256
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=optimsim
#SBATCH --output=optimsim.out
module load python/3.9.6 gurobi/9.5.0 mpi4py
source /home/kvolodin/python-venv/bin/activate
srun python3 optimSimulation.py