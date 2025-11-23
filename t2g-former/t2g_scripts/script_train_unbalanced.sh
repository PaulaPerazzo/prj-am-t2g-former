#!/bin/bash
#SBATCH --job-name=t2g_former_train_unbalanced
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH --cpus-per-task=16
#SBATCH -c 8
#SBATCH -o job.log
#SBATCH --output=job_output_train_unbalanced.txt
#SBATCH --error=job_error_train_unbalanced.txt
#SBATCH --partition=short-complex

# ativar ambiente
source /home/CIN/mpps/t2g-former/env/bin/activate

# executar •py
python run_t2g_unbalanced.py
