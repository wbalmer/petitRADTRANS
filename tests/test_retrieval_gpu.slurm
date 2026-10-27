#!/bin/bash
#SBATCH --job-name=prt4_jaxns  # Job name
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --account=b53080   #b1094           # allocation/account
#SBATCH --partition=buyin #ciera-gpu           # Partition or queue name, debug, shared, wholenode
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --gres=gpu:2                  # Specify the gpu and card
#SBATCH --mem-per-cpu=80G	      # set the amount of memory for the node
#SBATCH --time=1-00:00:00                  # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=balmer@northwestern.edu  # Email address for notifications

# get bashrc
source /home/efn6832/.bashrc

# Activate the virtual environment
mamba activate prt4

# Run your Python script
# chmod u+x test.sh
# ./test.sh

export OMP_NUM_THREADS=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
srun python run_test_retrieval.py

# Deactivate the virtual environment
conda deactivate

