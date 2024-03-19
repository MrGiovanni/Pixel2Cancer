# Pixel2Cancer

## Paper

<b>From Pixel to Cancer: Cellular Automata in Computed Tomography</b> <br/>
Yuxiang Lai<sup>1,2</sup>, Xiaoxi Chen<sup>2</sup>, Angtian Wang<sup>2</sup>, [Alan L. Yuille](https://www.cs.jhu.edu/~ayuille/)<sup>2</sup>, and [Zongwei Zhou](https://www.zongweiz.com/)<sup>2,*</sup> <br/>
<sup>1 </sup>Southeast University,  <br/>
<sup>2 </sup>Johns Hopkins University  <br/>
[paper](https://arxiv.org/abs/2403.06459) | [code](https://github.com/MrGiovanni/Pixel2Cancer/tree/main)

## Model
<div style="display:flex; justify-content:space-between;">

| Organ | Tumor | Model               | Pre-trained? | Download |          
|-------  |-------|---------------------|--------------|----------|
| liver | real  | unet                | no           | [link](#) |
| liver | real  | swin_unetrv2_base  | yes          | [link](#) |
| liver | real  | swin_unetrv2_base  | no           | [link](#) |
| liver | real  | swin_unetrv2_small | no           | [link](#) |
| liver | real  | swin_unetrv2_tiny  | no           | [link](#) |
| liver | synt  | unet                | no           | [link](#) |
| liver | synt  | swin_unetrv2_base  | yes          | [link](#) |
| liver | synt  | swin_unetrv2_base  | no           | [link](#) |
| liver | synt  | swin_unetrv2_small | no           | [link](#) |
| liver | synt  | swin_unetrv2_tiny  | no           | [link](#) |

| Organ | Tumor | Model               | Pre-trained? | Download |          
|-------  |-------|---------------------|--------------|----------|
| liver | real  | unet                | no           | [link](#) |
| liver | real  | swin_unetrv2_base  | yes          | [link](#) |
| liver | real  | swin_unetrv2_base  | no           | [link](#) |
| liver | real  | swin_unetrv2_small | no           | [link](#) |
| liver | real  | swin_unetrv2_tiny  | no           | [link](#) |
| liver | synt  | unet                | no           | [link](#) |
| liver | synt  | swin_unetrv2_base  | yes          | [link](#) |
| liver | synt  | swin_unetrv2_base  | no           | [link](#) |
| liver | synt  | swin_unetrv2_small | no           | [link](#) |
| liver | synt  | swin_unetrv2_tiny  | no           | [link](#) |

</div>

## 0. Installation

```bash
git clone https://github.com/MrGiovanni/Pixel2Cancer.git
# download pre-trained models
cd Pixel2Cancer/
wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt
```

See [installation instructions](INSTALL.md) to create an environment and obtain requirements.



## 1. Train segmentation models using synthetic tumors

```bash
datapath=/mnt/zzhou82/PublicAbdominalData/

# UNET (no.pretrain)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --dist-url=tcp://127.0.0.1:12235 --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/synt.no_pretrain.unet" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json

# Swin-UNETR-Base (pretrain)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --dist-url=tcp://127.0.0.1:12231 --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/synt.pretrain.swin_unetrv2_base" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json --use_pretrained

# Swin-UNETR-Base (no.pretrain)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --dist-url=tcp://127.0.0.1:12231 --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/synt.no_pretrain.swin_unetrv2_base" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json

# Swin-UNETR-Small (no.pretrain)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --dist-url=tcp://127.0.0.1:12233 --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/synt.no_pretrain.swin_unetrv2_small" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json

# Swin-UNETR-Tiny (no.pretrain)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --max_epochs=2000 --save_checkpoint --workers=0 --noamp --distributed --dist-url=tcp://127.0.0.1:12234 --cache_num=200 --val_overlap=0.5 --syn --logdir="runs/synt.no_pretrain.swin_unetrv2_tiny" --train_dir $datapath --val_dir $datapath --json_dir datafolds/healthy.json
```

## 2. Train segmentation models using real tumors (for comparison)
```bash
datapath=/mnt/zzhou82/PublicAbdominalData/

# UNET (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=unet --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12235 --cache_num=200 --logdir="runs/real.no_pretrain.unet" --train_dir $datapath --val_dir $datapath --json_dir datafolds/lits.json

# Swin-UNETR-Base (pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12231 --cache_num=200 --logdir="runs/real.pretrain.swin_unetrv2_base" --train_dir $datapath --val_dir $datapath --json_dir datafolds/lits.json --use_pretrained

# Swin-UNETR-Base (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=base --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12232 --cache_num=200 --logdir="runs/real.no_pretrain.swin_unetrv2_base" --train_dir $datapath --val_dir $datapath --json_dir datafolds/lits.json

# Swin-UNETR-Small (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=small --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12233 --cache_num=200 --logdir="runs/real.no_pretrain.swin_unetrv2_small" --train_dir $datapath --val_dir $datapath --json_dir datafolds/lits.json

# Swin-UNETR-Tiny (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore -W ignore main.py --optim_lr=4e-4 --batch_size=2 --lrschedule=warmup_cosine --optim_name=adamw --model_name=swin_unetrv2 --swin_type=tiny --val_every=200 --val_overlap=0.5 --max_epochs=2000 --save_checkpoint --workers=2 --noamp --distributed --dist-url=tcp://127.0.0.1:12234 --cache_num=200 --logdir="runs/real.no_pretrain.swin_unetrv2_tiny" --train_dir $datapath --val_dir $datapath --json_dir datafolds/lits.json
```
## 3. Evaluation

#### AI model trained by synthetic tumors
```bash
datapath=/mnt/zzhou82/PublicAbdominalData/

# UNET (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/synt.no_pretrain.unet --save_dir out
# Swin-UNETR-Base (pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/synt.pretrain.swin_unetrv2_base --save_dir out
# Swin-UNETR-Base (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/synt.no_pretrain.swin_unetrv2_base --save_dir out
# Swin-UNETR-Small (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/synt.no_pretrain.swin_unetrv2_small --save_dir out
# Swin-UNETR-Tiny (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/synt.no_pretrain.swin_unetrv2_tiny --save_dir out
```

#### AI model trained by real tumors
```bash
datapath=/mnt/zzhou82/PublicAbdominalData/

# UNET (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=unet --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/real.no_pretrain.unet --save_dir out
# Swin-UNETR-Base (pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/real.pretrain.swin_unetrv2_base --save_dir out
# Swin-UNETR-Base (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=base --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/real.no_pretrain.swin_unetrv2_base --save_dir out
# Swin-UNETR-Small (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=small --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/real.no_pretrain.swin_unetrv2_small --save_dir out
# Swin-UNETR-Tiny (no.pretrain)
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py --model=swin_unetrv2 --swin_type=tiny --val_overlap=0.75 --val_dir $datapath --json_dir datafolds/lits.json --log_dir runs/real.no_pretrain.swin_unetrv2_tiny --save_dir out
```

## 4. Data Setting
#### Train on 9k data of AbdominalAtlas1.1:
The release of AbdomenAtlas 1.0 can be found at [https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas_1.0_Mini](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas_1.0_Mini)
```bash
# AbdominalAtlas1.1 training data list
# Liver 
--json_dir /datafolds/Bodymap/Atlas9k_liver/Atlas9k_liver_fold0.json
--json_dir /datafolds/Bodymap/Atlas9k_liver/Atlas9k_liver_fold1.json
--json_dir /datafolds/Bodymap/Atlas9k_liver/Atlas9k_liver_fold2.json
--json_dir /datafolds/Bodymap/Atlas9k_liver/Atlas9k_liver_fold3.json
--json_dir /datafolds/Bodymap/Atlas9k_liver/Atlas9k_liver_fold4.json
#Pancreas
--json_dir /datafolds/Bodymap/Atlas9k_liver/Atlas9k_pancreas_fold0.json
--json_dir /datafolds/Bodymap/Atlas9k_liver/Atlas9k_pancreas_fold1.json
--json_dir /datafolds/Bodymap/Atlas9k_liver/Atlas9k_pancreas_fold2.json
--json_dir /datafolds/Bodymap/Atlas9k_liver/Atlas9k_pancreas_fold3.json
--json_dir /datafolds/Bodymap/Atlas9k_liver/Atlas9k_pancreas_fold4.json
#Kidney
--json_dir /datafolds/Bodymap/Atlas9k_liver/Atlas9k_kidney_fold0.json
--json_dir /datafolds/Bodymap/Atlas9k_liver/Atlas9k_kidney_fold1.json
--json_dir /datafolds/Bodymap/Atlas9k_liver/Atlas9k_kidney_fold2.json
--json_dir /datafolds/Bodymap/Atlas9k_liver/Atlas9k_kidney_fold3.json
--json_dir /datafolds/Bodymap/Atlas9k_liver/Atlas9k_kidney_fold4.json
```

#### Train on public data & Intern experiments:

```bash
# Public training data list
# Liver
--json_dir /datafolds/5_fold/liver/liver_tumor_0.json
--json_dir /datafolds/5_fold/liver/liver_tumor_1.json
--json_dir /datafolds/5_fold/liver/liver_tumor_2.json
--json_dir /datafolds/5_fold/liver/liver_tumor_3.json
--json_dir /datafolds/5_fold/liver/liver_tumor_4.json
# Pancreas
--json_dir /datafolds/5_fold/pancreas/pancreas_tumor_0.json
--json_dir /datafolds/5_fold/pancreas/pancreas_tumor_1.json
--json_dir /datafolds/5_fold/pancreas/pancreas_tumor_2.json
--json_dir /datafolds/5_fold/pancreas/pancreas_tumor_3.json
--json_dir /datafolds/5_fold/pancreas/pancreas_tumor_4.json
# Kidney
--json_dir /datafolds/5_fold/kidney/kidney_tumor_0.json
--json_dir /datafolds/5_fold/kidney/kidney_tumor_1.json
--json_dir /datafolds/5_fold/kidney/kidney_tumor_2.json
--json_dir /datafolds/5_fold/kidney/kidney_tumor_3.json
--json_dir /datafolds/5_fold/kidney/kidney_tumor_4.json
```



## Acknowledgement

This work was supported by the Lustgarten Foundation for Pancreatic Cancer Research and the McGovern Foundation. The segmentation backbone is based on [Swin UNETR](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_btcv_segmentation_3d.ipynb); we appreciate the effort of the [MONAI Team](https://monai.io/) to provide and maintain open-source code to the community.
