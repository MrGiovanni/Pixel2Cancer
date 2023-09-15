#!/bin/bash
#SBATCH --job-name=flare_validation

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -G a100:1
##SBATCH --exclusive
#SBATCH --mem=10G
#SBATCH -p general
#SBATCH -t 0-00:30:00
#SBATCH -q public

#SBATCH -o %x_slurm_%j.out     
#SBATCH -e %xslurm_%j.err      
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=zzhou82@asu.edu

module load mamba/latest
source activate pixel2cancer

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt

cd ./Tumor_Synthesis/cellular
python setup.py install
