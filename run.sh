echo $1 $2
sbatch --error=logs/$1.$2.out --output=logs/$1.$2.out "hg_$2_train_5_folds.sh" $1 $2

# ## 1. Train segmentation models using synthetic tumors
# for organ in kidney liver pancreas; do for fold in 1 2 3 4 5; do for config in synt.$organ.fold$fold.no_pretrain.unet synt.$organ.fold$fold.pretrain.swin_unetrv2_base synt.$organ.fold$fold.no_pretrain.swin_unetrv2_base synt.$organ.fold$fold.no_pretrain.swin_unetrv2_small synt.$organ.fold$fold.no_pretrain.swin_unetrv2_tiny; do bash run.sh $config $organ; done; done; done

# for organ in kidney; do for fold in 1 2; do for config in synt.$organ.fold$fold.no_pretrain.unet synt.$organ.fold$fold.pretrain.swin_unetrv2_base synt.$organ.fold$fold.no_pretrain.swin_unetrv2_base synt.$organ.fold$fold.no_pretrain.swin_unetrv2_small synt.$organ.fold$fold.no_pretrain.swin_unetrv2_tiny; do bash run.sh $config $organ; done; done; done

# ## 2. Train segmentation models using real tumors (for comparison)
# for config in real.no_pretrain.unet real.pretrain.swin_unetrv2_base real.no_pretrain.swin_unetrv2_base real.no_pretrain.swin_unetrv2_small real.no_pretrain.swin_unetrv2_tiny; do bash run.sh $config; done

# ## 3. Evaluate synt-tumor models using real tumors
# for config in eval.synt.no_pretrain.unet eval.synt.pretrain.swin_unetrv2_base eval.synt.no_pretrain.swin_unetrv2_base eval.synt.no_pretrain.swin_unetrv2_small eval.synt.no_pretrain.swin_unetrv2_tiny; do bash run.sh $config; done

# ## 4. Evaluate real-tumor models using real tumors
# for config in eval.real.no_pretrain.unet eval.real.pretrain.swin_unetrv2_base eval.real.no_pretrain.swin_unetrv2_base eval.real.no_pretrain.swin_unetrv2_small eval.real.no_pretrain.swin_unetrv2_tiny; do bash run.sh $config; done
