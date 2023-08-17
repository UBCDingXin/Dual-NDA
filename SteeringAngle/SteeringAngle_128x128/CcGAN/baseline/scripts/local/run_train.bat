::===============================================================
:: This is a batch script for running the program on windows 10! 
::===============================================================

@echo off

set METHOD_NAME=CcGAN_v3
set ROOT_PATH=C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/SteeringAngle/SteeringAngle_128x128/%METHOD_NAME%/baseline
set DATA_PATH=C:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/SteeringAngle
set EVAL_PATH=C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/SteeringAngle/SteeringAngle_128x128/evaluation/eval_models

set SEED=2023
set NUM_WORKERS=0
set MIN_LABEL=-80.0
set MAX_LABEL=80.0
set IMG_SIZE=128
set MAX_N_IMG_PER_LABEL=9999
set MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0

set BATCH_SIZE_G=128
set BATCH_SIZE_D=128
set NUM_D_STEPS=2
set SIGMA=-1.0
set KAPPA=-5.0
set LR_G=1e-4
set LR_D=1e-4
set NUM_ACC_D=2
set NUM_ACC_G=2

set GAN_ARCH="SAGAN"
set LOSS_TYPE="hinge"

set DIM_GAN=256
set DIM_EMBED=128

set niqe_dump_path="F:/LocalWD/CcGAN_TPAMI_NIQE/SteeringAngle/NIQE_128x128/fake_data"
set NITERS=20000
set Setting=niters20K
python main.py ^
    --setting_name %Setting% ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% --num_workers %NUM_WORKERS% ^
    --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% ^
    --max_num_img_per_label %MAX_N_IMG_PER_LABEL% --max_num_img_per_label_after_replica %MAX_N_IMG_PER_LABEL_AFTER_REPLICA% ^
    --GAN_arch %GAN_ARCH% --niters_gan %NITERS% --resume_niters_gan 0 --loss_type_gan %LOSS_TYPE% --num_D_steps %NUM_D_STEPS% ^
    --save_niters_freq 2000 --visualize_freq 2000 ^
    --batch_size_disc %BATCH_SIZE_D% --batch_size_gene %BATCH_SIZE_G% ^
    --lr_g %LR_G% --lr_d %LR_D% --dim_gan %DIM_GAN% --dim_embed %DIM_EMBED% ^
    --num_grad_acc_d %NUM_ACC_D% --num_grad_acc_g %NUM_ACC_G% ^
    --kernel_sigma %SIGMA% --threshold_type soft --kappa %KAPPA% ^
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout ^
    --comp_FID --samp_batch_size 500 --dump_fake_for_NIQE --niqe_dump_path %niqe_dump_path% ^ %*

    @REM --dump_fake_for_NIQE --niqe_dump_path %niqe_dump_path%


@REM set niqe_dump_path="F:/LocalWD/CcGAN_TPAMI_NIQE/SteeringAngle/NIQE_128x128/fake_data"
@REM set NITERS=20000
@REM Setting=niters20K
@REM python main.py ^
@REM     --setting_name %Setting% ^
@REM     --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% --num_workers %NUM_WORKERS% ^
@REM     --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% ^
@REM     --max_num_img_per_label %MAX_N_IMG_PER_LABEL% --max_num_img_per_label_after_replica %MAX_N_IMG_PER_LABEL_AFTER_REPLICA% ^
@REM     --GAN_arch %GAN_ARCH% --niters_gan %NITERS% --resume_niters_gan 0 --loss_type_gan %LOSS_TYPE% --num_D_steps %NUM_D_STEPS% ^
@REM     --save_niters_freq 2000 --visualize_freq 2000 ^
@REM     --batch_size_disc %BATCH_SIZE_D% --batch_size_gene %BATCH_SIZE_G% ^
@REM     --lr_g %LR_G% --lr_d %LR_D% --dim_gan %DIM_GAN% --dim_embed %DIM_EMBED% ^
@REM     --num_grad_acc_d %NUM_ACC_D% --num_grad_acc_g %NUM_ACC_G% ^
@REM     --kernel_sigma %SIGMA% --threshold_type soft --kappa %KAPPA% ^
@REM     --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout ^
@REM     --samp_batch_size 500 ^
@REM     --mae_filter --samp_filter_precnn_net vgg11 --samp_filter_mae_percentile_threshold 0.9 ^
@REM     --samp_filter_batch_size 200 --samp_filter_nfake_per_label_burnin 50 --samp_filter_nfake_per_label 50 ^
@REM     --niqe_filter --niqe_dump_path %niqe_dump_path% --niqe_nfake_per_label_burnin 50 ^ %*