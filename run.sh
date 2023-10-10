echo $1
sbatch --error=logs/$1.out --output=logs/$1.out hg.sh $1

# ## 1. Train segmentation models using synthetic tumors
# for config in synt.no_pretrain.unet synt.pretrain.swin_unetrv2_base synt.no_pretrain.swin_unetrv2_base synt.no_pretrain.swin_unetrv2_small synt.no_pretrain.swin_unetrv2_tiny; do bash run.sh $config; done

# ## 2. Train segmentation models using real tumors (for comparison)
# for config in real.no_pretrain.unet real.pretrain.swin_unetrv2_base real.no_pretrain.swin_unetrv2_base real.no_pretrain.swin_unetrv2_small real.no_pretrain.swin_unetrv2_tiny; do bash run.sh $config; done
