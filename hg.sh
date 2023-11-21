#!/bin/bash
#SBATCH --job-name=pixel2cancer

#SBATCH -N 1
#SBATCH -n 12
#SBATCH -G a100:2
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
# cd ./surface-distance
# pip install .
# cd ..

# cd ./Tumor_Synthesis/cellular
# python setup.py install

dist=$((RANDOM % 99999 + 10000))
datapath=/data/jliang12/zzhou82/datasets/PublicAbdominalData/

#Liver Augmentation

if [ $1 == 'mix_liver.no_pretrain.unet' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_liver/$1" --train_dir $datapath --val_dir $datapath --organ=liver --json_dir datafolds/mix_liver.json

elif [ $1 == 'mix_liver.pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_liver/$1" --train_dir $datapath --val_dir $datapath --organ=liver --json_dir datafolds/mix_liver.json --use_pretrained

elif [ $1 == 'mix_liver.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_liver/$1" --train_dir $datapath --val_dir $datapath --organ=liver --json_dir datafolds/mix_liver.json

elif [ $1 == 'mix_liver.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_liver/$1" --train_dir $datapath --val_dir $datapath --organ=liver --json_dir datafolds/mix_liver.json

elif [ $1 == 'mix_liver.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_liver/$1" --train_dir $datapath --val_dir $datapath --organ=liver --json_dir datafolds/mix_liver.json

elif [ $1 == 'eval.mix_liver.no_pretrain.unet' ]; then

    python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/mix_liver.json --log_dir runs_liver/synt.no_pretrain.unet --save_dir out

elif [ $1 == 'eval.mix_liver.pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/mix_liver.json --log_dir runs_liver/synt.pretrain.swin_unetrv2_base --save_dir out

elif [ $1 == 'eval.mix_liver.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/mix_liver.json --log_dir runs_liver/synt.no_pretrain.swin_unetrv2_base --save_dir out

elif [ $1 == 'eval.mix_liver.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/mix_liver.json --log_dir runs_liver/synt.no_pretrain.swin_unetrv2_small --save_dir out

elif [ $1 == 'eval.mix_liver.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/mix_liver.json --log_dir runs_liver/synt.no_pretrain.swin_unetrv2_tiny --save_dir out

fi

#Pancreas Augmentation

if [ $1 == 'mix_pancreas.no_pretrain.unet' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/mix_pancreas.json

elif [ $1 == 'mix_pancreas.pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/mix_pancreas.json --use_pretrained

elif [ $1 == 'mix_pancreas.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/mix_pancreas.json

elif [ $1 == 'mix_pancreas.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/mix_pancreas.json

elif [ $1 == 'mix_pancreas.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/mix_pancreas.json

elif [ $1 == 'eval.mix_pancreas.no_pretrain.unet' ]; then

    python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/mix_pancreas.json --log_dir runs_pancreas/synt.no_pretrain.unet --save_dir out

elif [ $1 == 'eval.mix_pancreas.pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/mix_pancreas.json --log_dir runs_pancreas/synt.pretrain.swin_unetrv2_base --save_dir out

elif [ $1 == 'eval.mix_pancreas.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/mix_pancreas.json --log_dir runs_pancreas/synt.no_pretrain.swin_unetrv2_base --save_dir out

elif [ $1 == 'eval.mix_pancreas.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/mix_pancreas.json --log_dir runs_pancreas/synt.no_pretrain.swin_unetrv2_small --save_dir out

elif [ $1 == 'eval.mix_pancreas.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/mix_pancreas.json --log_dir runs_pancreas/synt.no_pretrain.swin_unetrv2_tiny --save_dir out

fi

#Kidney Augmentation

if [ $1 == 'mix_kidney.no_pretrain.unet' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_kidney/$1" --train_dir $datapath --val_dir $datapath --organ=kidney --json_dir datafolds/mix_kidney.json

elif [ $1 == 'mix_kidney.pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_kidney/$1" --train_dir $datapath --val_dir $datapath --organ=kidney --json_dir datafolds/mix_kidney.json --use_pretrained

elif [ $1 == 'synt.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_kidney/$1" --train_dir $datapath --val_dir $datapath --organ=kidney --json_dir datafolds/mix_kidney.json

elif [ $1 == 'mix_kidney.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_kidney/$1" --train_dir $datapath --val_dir $datapath --organ=kidney --json_dir datafolds/mix_kidney.json

elif [ $1 == 'mix_kidney.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_kidney/$1" --train_dir $datapath --val_dir $datapath --organ=kidney --json_dir datafolds/mix_kidney.json

elif [ $1 == 'eval.mix_kidney.no_pretrain.unet' ]; then

    python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/mix_kidney.json --log_dir runs_kidney/synt.no_pretrain.unet --save_dir out

elif [ $1 == 'eval.mix_kidney.pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/mix_kidney.json --log_dir runs_kidney/synt.pretrain.swin_unetrv2_base --save_dir out

elif [ $1 == 'eval.mix_kidney.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/mix_kidney.json --log_dir runs_kidney/synt.no_pretrain.swin_unetrv2_base --save_dir out

elif [ $1 == 'eval.mix_kidney.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/mix_kidney.json --log_dir runs_kidney/synt.no_pretrain.swin_unetrv2_small --save_dir out

elif [ $1 == 'eval.mix_kidney.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/mix_kidney.json --log_dir runs_kidney/synt.no_pretrain.swin_unetrv2_tiny --save_dir out

fi


#Kidney

# if [ $1 == 'synt.no_pretrain.unet' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=kidney --json_dir datafolds/healthy_kidney.json

# elif [ $1 == 'synt.pretrain.swin_unetrv2_base' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=kidney --json_dir datafolds/healthy_kidney.json --use_pretrained

# elif [ $1 == 'synt.no_pretrain.swin_unetrv2_base' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=kidney --json_dir datafolds/healthy_kidney.json

# elif [ $1 == 'synt.no_pretrain.swin_unetrv2_small' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=kidney --json_dir datafolds/healthy_kidney.json

# elif [ $1 == 'synt.no_pretrain.swin_unetrv2_tiny' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=kidney --json_dir datafolds/real_kidney.json

# elif [ $1 == 'real.no_pretrain.unet' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=kidney --json_dir datafolds/real_kidney.json

# elif [ $1 == 'real.pretrain.swin_unetrv2_base' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=kidney --json_dir datafolds/real_kidney.json --use_pretrained

# elif [ $1 == 'real.no_pretrain.swin_unetrv2_base' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=kidney --json_dir datafolds/real_kidney.json

# elif [ $1 == 'real.no_pretrain.swin_unetrv2_small' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=kidney --json_dir datafolds/real_kidney.json

# elif [ $1 == 'real.no_pretrain.swin_unetrv2_tiny' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=kidney --json_dir datafolds/real_kidney.json

# elif [ $1 == 'eval.synt.no_pretrain.unet' ]; then

#     python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/real_kidney.json --log_dir runs/synt.no_pretrain.unet --save_dir out

# elif [ $1 == 'eval.synt.pretrain.swin_unetrv2_base' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/real_kidney.json --log_dir runs/synt.pretrain.swin_unetrv2_base --save_dir out

# elif [ $1 == 'eval.synt.no_pretrain.swin_unetrv2_base' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/real_kidney.json --log_dir runs/synt.no_pretrain.swin_unetrv2_base --save_dir out

# elif [ $1 == 'eval.synt.no_pretrain.swin_unetrv2_small' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/real_kidney.json --log_dir runs/synt.no_pretrain.swin_unetrv2_small --save_dir out

# elif [ $1 == 'eval.synt.no_pretrain.swin_unetrv2_tiny' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/real_kidney.json --log_dir runs/synt.no_pretrain.swin_unetrv2_tiny --save_dir out

# elif [ $1 == 'eval.real.no_pretrain.unet' ]; then

#     python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/real_kidney.json --log_dir runs/real.no_pretrain.unet --save_dir out

# elif [ $1 == 'eval.real.pretrain.swin_unetrv2_base' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/real_kidney.json --log_dir runs/real.pretrain.swin_unetrv2_base --save_dir out

# elif [ $1 == 'eval.real.no_pretrain.swin_unetrv2_base' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/real_kidney.json --log_dir runs/real.no_pretrain.swin_unetrv2_base --save_dir out

# elif [ $1 == 'eval.real.no_pretrain.swin_unetrv2_small' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/real_kidney.json --log_dir runs/real.no_pretrain.swin_unetrv2_small --save_dir out

# elif [ $1 == 'eval.real.no_pretrain.swin_unetrv2_tiny' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/real_kidney.json --log_dir runs/real.no_pretrain.swin_unetrv2_tiny --save_dir out

# fi



# Pancreas

# if [ $1 == 'synt.no_pretrain.unet' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/healthy_pancreas.json

# elif [ $1 == 'synt.pretrain.swin_unetrv2_base' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/healthy_pancreas.json --use_pretrained

# elif [ $1 == 'synt.no_pretrain.swin_unetrv2_base' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/healthy_pancreas.json

# elif [ $1 == 'synt.no_pretrain.swin_unetrv2_small' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/healthy_pancreas.json

# elif [ $1 == 'synt.no_pretrain.swin_unetrv2_tiny' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/healthy_pancreas.json

# elif [ $1 == 'real.no_pretrain.unet' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/real_pancreas.json

# elif [ $1 == 'real.pretrain.swin_unetrv2_base' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/real_pancreas.json --use_pretrained

# elif [ $1 == 'real.no_pretrain.swin_unetrv2_base' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/real_pancreas.json

# elif [ $1 == 'real.no_pretrain.swin_unetrv2_small' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/real_pancreas.json

# elif [ $1 == 'real.no_pretrain.swin_unetrv2_tiny' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/real_pancreas.json

# elif [ $1 == 'eval.synt.no_pretrain.unet' ]; then

#     python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/healthy_pancreas.json --log_dir runs/synt.no_pretrain.unet --save_dir out

# elif [ $1 == 'eval.synt.pretrain.swin_unetrv2_base' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/healthy_pancreas.json --log_dir runs/synt.pretrain.swin_unetrv2_base --save_dir out

# elif [ $1 == 'eval.synt.no_pretrain.swin_unetrv2_base' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/healthy_pancreas.json --log_dir runs/synt.no_pretrain.swin_unetrv2_base --save_dir out

# elif [ $1 == 'eval.synt.no_pretrain.swin_unetrv2_small' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/healthy_pancreas.json --log_dir runs/synt.no_pretrain.swin_unetrv2_small --save_dir out

# elif [ $1 == 'eval.synt.no_pretrain.swin_unetrv2_tiny' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/healthy_pancreas.json --log_dir runs/synt.no_pretrain.swin_unetrv2_tiny --save_dir out

# elif [ $1 == 'eval.real.no_pretrain.unet' ]; then

#     python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/real_pancreas.json --log_dir runs/real.no_pretrain.unet --save_dir out

# elif [ $1 == 'eval.real.pretrain.swin_unetrv2_base' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/real_pancreas.json --log_dir runs/real.pretrain.swin_unetrv2_base --save_dir out

# elif [ $1 == 'eval.real.no_pretrain.swin_unetrv2_base' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/real_pancreas.json --log_dir runs/real.no_pretrain.swin_unetrv2_base --save_dir out

# elif [ $1 == 'eval.real.no_pretrain.swin_unetrv2_small' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/real_pancreas.json --log_dir runs/real.no_pretrain.swin_unetrv2_small --save_dir out

# elif [ $1 == 'eval.real.no_pretrain.swin_unetrv2_tiny' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/real_pancreas.json --log_dir runs/real.no_pretrain.swin_unetrv2_tiny --save_dir out

# fi

# Liver

# if [ $1 == 'synt.no_pretrain.unet' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json

# elif [ $1 == 'synt.pretrain.swin_unetrv2_base' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json --use_pretrained

# elif [ $1 == 'synt.no_pretrain.swin_unetrv2_base' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json

# elif [ $1 == 'synt.no_pretrain.swin_unetrv2_small' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json

# elif [ $1 == 'synt.no_pretrain.swin_unetrv2_tiny' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json

# elif [ $1 == 'real.no_pretrain.unet' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --json_dir datafolds/lits.json

# elif [ $1 == 'real.pretrain.swin_unetrv2_base' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --json_dir datafolds/lits.json --use_pretrained

# elif [ $1 == 'real.no_pretrain.swin_unetrv2_base' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --json_dir datafolds/lits.json

# elif [ $1 == 'real.no_pretrain.swin_unetrv2_small' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --json_dir datafolds/lits.json

# elif [ $1 == 'real.no_pretrain.swin_unetrv2_tiny' ]; then

#     python -W ignore main.py --optim_lr=4e-4 --batch_size=8 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --logdir="runs/$1" --train_dir $datapath --val_dir $datapath --json_dir datafolds/lits.json

# elif [ $1 == 'eval.synt.no_pretrain.unet' ]; then

#     python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/synt.no_pretrain.unet --save_dir out

# elif [ $1 == 'eval.synt.pretrain.swin_unetrv2_base' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/synt.pretrain.swin_unetrv2_base --save_dir out

# elif [ $1 == 'eval.synt.no_pretrain.swin_unetrv2_base' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/synt.no_pretrain.swin_unetrv2_base --save_dir out

# elif [ $1 == 'eval.synt.no_pretrain.swin_unetrv2_small' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/synt.no_pretrain.swin_unetrv2_small --save_dir out

# elif [ $1 == 'eval.synt.no_pretrain.swin_unetrv2_tiny' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/synt.no_pretrain.swin_unetrv2_tiny --save_dir out

# elif [ $1 == 'eval.real.no_pretrain.unet' ]; then

#     python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/real.no_pretrain.unet --save_dir out

# elif [ $1 == 'eval.real.pretrain.swin_unetrv2_base' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/real.pretrain.swin_unetrv2_base --save_dir out

# elif [ $1 == 'eval.real.no_pretrain.swin_unetrv2_base' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/real.no_pretrain.swin_unetrv2_base --save_dir out

# elif [ $1 == 'eval.real.no_pretrain.swin_unetrv2_small' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/real.no_pretrain.swin_unetrv2_small --save_dir out

# elif [ $1 == 'eval.real.no_pretrain.swin_unetrv2_tiny' ]; then

#     python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/real.no_pretrain.swin_unetrv2_tiny --save_dir out

# fi
