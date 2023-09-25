#!/bin/bash
#SBATCH --job-name=pixel2cancer

#SBATCH -N 1
#SBATCH -n 8
#SBATCH -G a100:1
##SBATCH --exclusive
#SBATCH --mem=150G
#SBATCH -p general
#SBATCH -t 7-00:00:00
#SBATCH -q public

#SBATCH -o %x_slurm_%j.out     
#SBATCH -e %xslurm_%j.err      
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=zzhou82@asu.edu

module load mamba/latest
source activate pixel2cancer

# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# pip install -r requirements.txt

# cd ./Tumor_Synthesis/cellular
# python setup.py install

dist=$((RANDOM % 99999 + 10000))
datapath=/data/jliang12/zzhou82/datasets/PublicAbdominalData/

## 1. Train segmentation models using synthetic tumors

# UNET (no.pretrain)
python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/synt.no_pretrain.unet" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json
# sbatch --error=logs/synt.no_pretrain.unet.out --output=logs/synt.no_pretrain.unet.out hg.sh

# Swin-UNETR-Base (pretrain)
python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/synt.pretrain.swin_unetrv2_base" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json --use_pretrained
# sbatch --error=logs/synt.pretrain.swin_unetrv2_base.out --output=logs/synt.pretrain.swin_unetrv2_base.out hg.sh

# Swin-UNETR-Base (no.pretrain)
python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/synt.no_pretrain.swin_unetrv2_base" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json
# sbatch --error=logs/synt.no_pretrain.swin_unetrv2_base.out --output=logs/synt.no_pretrain.swin_unetrv2_base.out hg.sh

# Swin-UNETR-Small (no.pretrain)
python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/synt.no_pretrain.swin_unetrv2_small" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json
# sbatch --error=logs/synt.no_pretrain.swin_unetrv2_small.out --output=logs/synt.no_pretrain.swin_unetrv2_small.out hg.sh

# Swin-UNETR-Tiny (no.pretrain)
python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/synt.no_pretrain.swin_unetrv2_tiny" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json
# sbatch --error=logs/synt.no_pretrain.swin_unetrv2_tiny.out --output=logs/synt.no_pretrain.swin_unetrv2_tiny.out hg.sh

## 2. Train segmentation models using real tumors (for comparison)

# UNET (no.pretrain)
python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/real.no_pretrain.unet" --train_dir $datapath --val_dir $datapath --json_dir datafolds/lits.json
# sbatch --error=logs/real.no_pretrain.unet.out --output=logs/real.no_pretrain.unet.out hg.sh

# Swin-UNETR-Base (pretrain)
python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/real.pretrain.swin_unetrv2_base" --train_dir $datapath --val_dir $datapath --json_dir datafolds/lits.json --use_pretrained
# sbatch --error=logs/real.pretrain.swin_unetrv2_base.out --output=logs/real.pretrain.swin_unetrv2_base.out hg.sh

# Swin-UNETR-Base (no.pretrain)
python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/real.no_pretrain.swin_unetrv2_base" --train_dir $datapath --val_dir $datapath --json_dir datafolds/lits.json
# sbatch --error=logs/real.no_pretrain.swin_unetrv2_base.out --output=logs/real.no_pretrain.swin_unetrv2_base.out hg.sh

# Swin-UNETR-Small (no.pretrain)
python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/real.no_pretrain.swin_unetrv2_small" --train_dir $datapath --val_dir $datapath --json_dir datafolds/lits.json
# sbatch --error=logs/real.no_pretrain.swin_unetrv2_small.out --output=logs/real.no_pretrain.swin_unetrv2_small.out hg.sh

# Swin-UNETR-Tiny (no.pretrain)
python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/real.no_pretrain.swin_unetrv2_tiny" --train_dir $datapath --val_dir $datapath --json_dir datafolds/lits.json
# sbatch --error=logs/real.no_pretrain.swin_unetrv2_tiny.out --output=logs/real.no_pretrain.swin_unetrv2_tiny.out hg.sh
