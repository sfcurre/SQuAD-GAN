#!/bin/bash
#SBATCH --job-name=SQuAD_GAN
#SBATCH --time=47:59:59
#SBATCH --output="SQuAD_GAN_train-%j.out"
#SBATCH --account=PAS1449
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL

set -x
set -e

cd /users/PAS0166/sfcurre

source /usr/local/python/3.6-conda5.2/etc/profile.d/conda.sh
conda activate deepml4
python squan_gan.py -j train-v2.0.json -e 10
