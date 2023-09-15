# Pixel2Cancer

## 0.Preliminary

##### Create environment
```bash
module load mamba/latest # only for Sol
mamba create -n pixel2cancer python=3.9
```

##### Installation
```
git clone https://github.com/MrGiovanni/Pixel2Cancer.git

# install requirements
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt

# install cellular automata
cd ./Tumor_Synthesis/cellular
python setup.py install 

```

## 1.Train segmentation models using synthetic tumors

```
datapath=/mnt/zzhou82/PublicAbdominalData/

# UNET (no.pretrain)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=4000 --save_checkpoint --workers=0 --noamp --distributed --dist-url=tcp://127.0.0.1:12235 --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/synt.no_pretrain.unet" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json
# Swin-UNETR-Base (pretrain)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=4000 --save_checkpoint --workers=0 --noamp --distributed --dist-url=tcp://127.0.0.1:12231 --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/synt.pretrain.swin_unetrv2_base" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json --use_pretrained
# Swin-UNETR-Base (no.pretrain)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=4000 --save_checkpoint --workers=0 --noamp --distributed --dist-url=tcp://127.0.0.1:12231 --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/synt.no_pretrain.swin_unetrv2_base" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json
# Swin-UNETR-Small (no.pretrain)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=4000 --save_checkpoint --workers=0 --noamp --distributed --dist-url=tcp://127.0.0.1:12233 --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/synt.no_pretrain.swin_unetrv2_small" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json
# Swin-UNETR-Tiny (no.pretrain)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=4000 --save_checkpoint --workers=0 --noamp --distributed --dist-url=tcp://127.0.0.1:12234 --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/synt.no_pretrain.swin_unetrv2_tiny" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json

```
