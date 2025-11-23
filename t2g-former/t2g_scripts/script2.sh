#!/bin/bash
#SBATCH --job-name=t2g_former_optuna_not_balanced_data
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH --cpus-per-task=16
#SBATCH -c 8
#SBATCH -o job.log
#SBATCH --output=job_output_2.txt
#SBATCH --error=job_error_2.txt
#SBATCH --partition=short-simple

# ativar ambiente
source /home/CIN/mpps/t2g-former/env/bin/activate

# executar •py
python tune_t2g_for_unbalanced_data.py
