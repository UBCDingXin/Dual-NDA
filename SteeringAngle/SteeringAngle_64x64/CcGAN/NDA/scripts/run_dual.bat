::===============================================================
:: This is a batch script for running the program on windows! 
:: Please set the following path correctly! 
:: Recommend using absolute path!
::===============================================================

@echo off

set METHOD_NAME=CcGAN
set ROOT_PREFIX=<Your_Path>/Dual-NDA/SteeringAngle/SteeringAngle_64x64
set ROOT_PATH=%ROOT_PREFIX%/%METHOD_NAME%/NDA
set DATA_PATH=<Your_Path>/Dual-NDA/datasets/SteeringAngle
set EVAL_PATH=%ROOT_PREFIX%/evaluation/eval_models
set niqe_dump_path=<Your_Path>/Dual-NDA/NIQE/SteeringAngle/NIQE_64x64/fake_data

set SEED=2023
set NUM_WORKERS=0
set MIN_LABEL=-80.0
set MAX_LABEL=80.0
set IMG_SIZE=64
set MAX_N_IMG_PER_LABEL=9999
set MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0

set BATCH_SIZE_G=512
set BATCH_SIZE_D=512
set NUM_D_STEPS=2
set SIGMA=-1.0
set KAPPA=-5.0
set LR_G=1e-4
set LR_D=1e-4
set NUM_ACC_D=1
set NUM_ACC_G=1

set GAN_ARCH="SAGAN"
set LOSS_TYPE="hinge"
set DIM_GAN=256
set DIM_EMBED=128


set SETTING="Setup1"

set fake_data_path_1=%ROOT_PREFIX%/%METHOD_NAME%/baseline/output/SAGAN_soft_si0.029_ka1000.438_hinge_nDs2_nDa1_nGa1_Dbs512_Gbs512/bad_fake_data/niters20K/badfake_NIQE0.9_nfake17740.h5
set nda_c_quantile=0.5
set nda_start_iter=0

set NITERS=20000
set resume_niter=0
python main.py ^
    --setting_name %SETTING% --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% --num_workers %NUM_WORKERS% ^
    --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% ^
    --max_num_img_per_label %MAX_N_IMG_PER_LABEL% --max_num_img_per_label_after_replica %MAX_N_IMG_PER_LABEL_AFTER_REPLICA% ^
    --GAN_arch %GAN_ARCH% --niters_gan %NITERS% --resume_niters_gan %resume_niter% --loss_type_gan %LOSS_TYPE% --num_D_steps %NUM_D_STEPS% ^
    --save_niters_freq 2500 --visualize_freq 500 ^
    --batch_size_disc %BATCH_SIZE_D% --batch_size_gene %BATCH_SIZE_G% ^
    --num_grad_acc_d %NUM_ACC_D% --num_grad_acc_g %NUM_ACC_G% ^
    --lr_g %LR_G% --lr_d %LR_D% --dim_gan %DIM_GAN% --dim_embed %DIM_EMBED% ^
    --kernel_sigma %SIGMA% --threshold_type soft --kappa %KAPPA% ^
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout ^
    --nda_start_iter %nda_start_iter% ^
    --nda_a 0.7 --nda_b 0 --nda_c 0.1 --nda_d 0.2 --nda_e 0 --nda_c_quantile %nda_c_quantile% ^
    --path2badfake1 %fake_data_path_1% ^
    --comp_FID --samp_batch_size 500 --dump_fake_for_NIQE --niqe_dump_path %niqe_dump_path% ^ %*