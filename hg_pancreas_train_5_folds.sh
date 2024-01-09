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


#pancreas Real



#pancreas Synthesis

if [ $1 == 'aug_pancreas_0.no_pretrain.unet' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json

elif [ $1 == 'aug_pancreas_0.pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json --use_pretrained

elif [ $1 == 'aug_pancreas_0.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json

elif [ $1 == 'aug_pancreas_0.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json

elif [ $1 == 'aug_pancreas_0.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json

elif [ $1 == 'aug_pancreas_1.no_pretrain.unet' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json

elif [ $1 == 'aug_pancreas_1.pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json --use_pretrained

elif [ $1 == 'aug_pancreas_1.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json

elif [ $1 == 'aug_pancreas_1.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json

elif [ $1 == 'aug_pancreas_1.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json

elif [ $1 == 'aug_pancreas_2.no_pretrain.unet' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json

elif [ $1 == 'aug_pancreas_2.pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json --use_pretrained

elif [ $1 == 'aug_pancreas_2.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json

elif [ $1 == 'aug_pancreas_2.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json

elif [ $1 == 'aug_pancreas_2.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json

elif [ $1 == 'aug_pancreas_3.no_pretrain.unet' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json

elif [ $1 == 'aug_pancreas_3.pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json --use_pretrained

elif [ $1 == 'aug_pancreas_3.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json

elif [ $1 == 'aug_pancreas_3.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json

elif [ $1 == 'aug_pancreas_3.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json
        
elif [ $1 == 'aug_pancreas_4.no_pretrain.unet' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json

elif [ $1 == 'aug_pancreas_4.pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json --use_pretrained

elif [ $1 == 'aug_pancreas_4.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json

elif [ $1 == 'aug_pancreas_4.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json

elif [ $1 == 'aug_pancreas_4.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=2 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5 --syn --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json
    
fi


#pancreas Real

if [ $1 == 'real_pancreas_0.no_pretrain.unet' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json

elif [ $1 == 'real_pancreas_0.pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json --use_pretrained

elif [ $1 == 'real_pancreas_0.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json

elif [ $1 == 'real_pancreas_0.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json

elif [ $1 == 'real_pancreas_0.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json

elif [ $1 == 'real_pancreas_1.no_pretrain.unet' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json

elif [ $1 == 'real_pancreas_1.pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json --use_pretrained

elif [ $1 == 'real_pancreas_1.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json

elif [ $1 == 'real_pancreas_1.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json

elif [ $1 == 'real_pancreas_1.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json

elif [ $1 == 'real_pancreas_2.no_pretrain.unet' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json

elif [ $1 == 'real_pancreas_2.pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json --use_pretrained

elif [ $1 == 'real_pancreas_2.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json

elif [ $1 == 'real_pancreas_2.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json

elif [ $1 == 'real_pancreas_2.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json

elif [ $1 == 'real_pancreas_3.no_pretrain.unet' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json

elif [ $1 == 'real_pancreas_3.pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json --use_pretrained

elif [ $1 == 'real_pancreas_3.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json

elif [ $1 == 'real_pancreas_3.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json

elif [ $1 == 'real_pancreas_3.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json
        
elif [ $1 == 'real_pancreas_4.no_pretrain.unet' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json

elif [ $1 == 'real_pancreas_4.pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json --use_pretrained

elif [ $1 == 'real_pancreas_4.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json

elif [ $1 == 'real_pancreas_4.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json

elif [ $1 == 'real_pancreas_4.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore main.py --optim_lr=4e-4 --batch_size=4 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --ngpus_per_node=1 --dist-url=tcp://127.0.0.1:$dist --cache_num=200 --val_overlap=0.5  --logdir="runs_pancreas/$1" --train_dir $datapath --val_dir $datapath --organ=pancreas --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json
    
fi