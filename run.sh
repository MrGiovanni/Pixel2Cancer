echo $1 $2 $3
sbatch --error=logs/$1.$2.$3.out --output=logs/$1.$2.$3.out "hg_$2_$3_5_folds.sh" $1 $2 $3

# ## 1. Train segmentation models using synthetic tumors
# for organ in kidney liver pancreas; do for fold in 0 1 2 3 4; do for config in synt.$organ.fold$fold.no_pretrain.unet synt.$organ.fold$fold.pretrain.swin_unetrv2_base synt.$organ.fold$fold.no_pretrain.swin_unetrv2_base synt.$organ.fold$fold.no_pretrain.swin_unetrv2_small synt.$organ.fold$fold.no_pretrain.swin_unetrv2_tiny; do bash run.sh $config $organ train; done; done; done

# for organ in kidney liver pancreas; do for fold in 0 1; do for config in synt.$organ.fold$fold.no_pretrain.unet synt.$organ.fold$fold.pretrain.swin_unetrv2_base synt.$organ.fold$fold.no_pretrain.swin_unetrv2_base synt.$organ.fold$fold.no_pretrain.swin_unetrv2_small synt.$organ.fold$fold.no_pretrain.swin_unetrv2_tiny; do bash run.sh $config $organ train; done; done; done

# ## 2. Train segmentation models using real tumors (for comparison)
# for organ in kidney liver pancreas; do for fold in 0 1 2 3 4; do for config in real.$organ.fold$fold.no_pretrain.unet real.$organ.fold$fold.pretrain.swin_unetrv2_base real.$organ.fold$fold.no_pretrain.swin_unetrv2_base real.$organ.fold$fold.no_pretrain.swin_unetrv2_small real.$organ.fold$fold.no_pretrain.swin_unetrv2_tiny; do bash run.sh $config $organ train; done; done; done

# ## 3. Evaluate synt-tumor models using real tumors
# for organ in kidney liver pancreas; do for fold in 0 1 2 3 4; do for config in eval.synt.$organ.fold$fold.no_pretrain.unet eval.synt.$organ.fold$fold.pretrain.swin_unetrv2_base eval.synt.$organ.fold$fold.no_pretrain.swin_unetrv2_base eval.synt.$organ.fold$fold.no_pretrain.swin_unetrv2_small eval.synt.$organ.fold$fold.no_pretrain.swin_unetrv2_tiny; do bash run.sh $config $organ eval; done; done; done

# ## 4. Evaluate real-tumor models using real tumors
# for organ in kidney liver pancreas; do for fold in 0 1 2 3 4; do for config in eval.real.$organ.fold$fold.no_pretrain.unet eval.real.$organ.fold$fold.pretrain.swin_unetrv2_base eval.real.$organ.fold$fold.no_pretrain.swin_unetrv2_base eval.real.$organ.fold$fold.no_pretrain.swin_unetrv2_small eval.real.$organ.fold$fold.no_pretrain.swin_unetrv2_tiny; do bash run.sh $config $organ eval; done; done; done
