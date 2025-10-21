#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --time 30:00
#SBATCH -p exercise-eml
#SBATCH -o my-job-output

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

python template.py
