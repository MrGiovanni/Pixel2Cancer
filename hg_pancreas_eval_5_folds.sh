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

#SBATCH -o %x_slurm_%j.out/pancreas     
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

if [ $1 == 'eval.aug_pancreas_0.no_pretrain.unet' ]; then

    python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json --log_dir runs_pancreas/aug_pancreas_0.no_pretrain.unet --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_0.pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json --log_dir runs_pancreas/aug_pancreas_0.pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_0.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json --log_dir runs_pancreas/aug_pancreas_0.no_pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_0.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json --log_dir runs_pancreas/aug_pancreas_0.no_pretrain.swin_unetrv2_small --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_0.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json --log_dir runs_pancreas/aug_pancreas_0.no_pretrain.swin_unetrv2_tiny --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_1.no_pretrain.unet' ]; then

    python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json --log_dir runs_pancreas/aug_pancreas_1.no_pretrain.unet --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_1.pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json --log_dir runs_pancreas/aug_pancreas_1.pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_1.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json --log_dir runs_pancreas/aug_pancreas_1.no_pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_1.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json --log_dir runs_pancreas/aug_pancreas_1.no_pretrain.swin_unetrv2_small --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_1.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json --log_dir runs_pancreas/aug_pancreas_1.no_pretrain.swin_unetrv2_tiny --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_2.no_pretrain.unet' ]; then

    python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json --log_dir runs_pancreas/aug_pancreas_2.no_pretrain.unet --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_2.pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json --log_dir runs_pancreas/aug_pancreas_2.pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_2.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json --log_dir runs_pancreas/aug_pancreas_2.no_pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_2.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json --log_dir runs_pancreas/aug_pancreas_2.no_pretrain.swin_unetrv2_small --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_2.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json --log_dir runs_pancreas/aug_pancreas_2.no_pretrain.swin_unetrv2_tiny --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_3.no_pretrain.unet' ]; then

    python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json --log_dir runs_pancreas/aug_pancreas_3.no_pretrain.unet --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_3.pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json --log_dir runs_pancreas/aug_pancreas_3.pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_3.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json --log_dir runs_pancreas/aug_pancreas_3.no_pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_3.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json --log_dir runs_pancreas/aug_pancreas_3.no_pretrain.swin_unetrv2_small --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_3.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json --log_dir runs_pancreas/aug_pancreas_3.no_pretrain.swin_unetrv2_tiny --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_4.no_pretrain.unet' ]; then

    python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json --log_dir runs_pancreas/aug_pancreas_4.no_pretrain.unet --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_4.pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json --log_dir runs_pancreas/aug_pancreas_4.pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_4.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json --log_dir runs_pancreas/aug_pancreas_4.no_pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_4.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json --log_dir runs_pancreas/aug_pancreas_4.no_pretrain.swin_unetrv2_small --save_dir out/pancreas

elif [ $1 == 'eval.aug_pancreas_4.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json --log_dir runs_pancreas/aug_pancreas_4.no_pretrain.swin_unetrv2_tiny --save_dir out/pancreas   

fi



# real


if [ $1 == 'eval.real_pancreas_0.no_pretrain.unet' ]; then

    python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json --log_dir runs_pancreas/real_pancreas_0.no_pretrain.unet --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_0.pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json --log_dir runs_pancreas/real_pancreas_0.pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_0.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json --log_dir runs_pancreas/real_pancreas_0.no_pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_0.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json --log_dir runs_pancreas/real_pancreas_0.no_pretrain.swin_unetrv2_small --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_0.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_0.json --log_dir runs_pancreas/real_pancreas_0.no_pretrain.swin_unetrv2_tiny --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_1.no_pretrain.unet' ]; then

    python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json --log_dir runs_pancreas/real_pancreas_1.no_pretrain.unet --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_1.pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json --log_dir runs_pancreas/real_pancreas_1.pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_1.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json --log_dir runs_pancreas/real_pancreas_1.no_pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_1.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json --log_dir runs_pancreas/real_pancreas_1.no_pretrain.swin_unetrv2_small --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_1.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_1.json --log_dir runs_pancreas/real_pancreas_1.no_pretrain.swin_unetrv2_tiny --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_2.no_pretrain.unet' ]; then

    python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json --log_dir runs_pancreas/real_pancreas_2.no_pretrain.unet --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_2.pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json --log_dir runs_pancreas/real_pancreas_2.pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_2.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json --log_dir runs_pancreas/real_pancreas_2.no_pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_2.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json --log_dir runs_pancreas/real_pancreas_2.no_pretrain.swin_unetrv2_small --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_2.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_2.json --log_dir runs_pancreas/real_pancreas_2.no_pretrain.swin_unetrv2_tiny --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_3.no_pretrain.unet' ]; then

    python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json --log_dir runs_pancreas/real_pancreas_3.no_pretrain.unet --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_3.pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json --log_dir runs_pancreas/real_pancreas_3.pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_3.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json --log_dir runs_pancreas/real_pancreas_3.no_pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_3.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json --log_dir runs_pancreas/real_pancreas_3.no_pretrain.swin_unetrv2_small --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_3.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_3.json --log_dir runs_pancreas/real_pancreas_3.no_pretrain.swin_unetrv2_tiny --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_4.no_pretrain.unet' ]; then

    python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json --log_dir runs_pancreas/real_pancreas_4.no_pretrain.unet --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_4.pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json --log_dir runs_pancreas/real_pancreas_4.pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_4.no_pretrain.swin_unetrv2_base' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json --log_dir runs_pancreas/real_pancreas_4.no_pretrain.swin_unetrv2_base --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_4.no_pretrain.swin_unetrv2_small' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json --log_dir runs_pancreas/real_pancreas_4.no_pretrain.swin_unetrv2_small --save_dir out/pancreas

elif [ $1 == 'eval.real_pancreas_4.no_pretrain.swin_unetrv2_tiny' ]; then

    python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/5_fold/pancreas/pancreas_tumor_4.json --log_dir runs_pancreas/real_pancreas_4.no_pretrain.swin_unetrv2_tiny --save_dir out/pancreas   

fi