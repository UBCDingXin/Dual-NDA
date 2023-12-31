::===============================================================
:: This is a batch script for running the program on windows! 
:: Please set the following path correctly! 
:: Recommend using absolute path!
::===============================================================

@echo off

set METHOD_NAME=CcGAN
set ROOT_PREFIX=<Your_Path>/Dual-NDA/UTKFace/UTKFace_64x64
set ROOT_PATH=%ROOT_PREFIX%/%METHOD_NAME%/NDA
set DATA_PATH=<Your_Path>/Dual-NDA/datasets/UTKFace
set EVAL_PATH=%ROOT_PREFIX%/evaluation/eval_models
set dump_niqe_path=<Your_Path>/Dual-NDA/NIQE/UTKFace/NIQE_64x64/fake_data

set SEED=2023
set NUM_WORKERS=0
set MIN_LABEL=1
set MAX_LABEL=60
set IMG_SIZE=64
set MAX_N_IMG_PER_LABEL=2000
set MAX_N_IMG_PER_LABEL_AFTER_REPLICA=200

set BATCH_SIZE_G=256
set BATCH_SIZE_D=256
set NUM_D_STEPS=2
set SIGMA=-1.0
set KAPPA=-1.0
set LR_G=1e-4
set LR_D=1e-4
set NUM_ACC_D=1
set NUM_ACC_G=1

set GAN_ARCH=SNGAN
set LOSS_TYPE=vanilla
set DIM_GAN=256
set DIM_EMBED=128


set SETTING="Setup_vNDA"

set nda_c_quantile=0.9
set nda_start_iter=0

set NITERS=40000
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
    --nda_a 0.25 --nda_b 0.75 --nda_c 0 --nda_d 0 --nda_e 0 --nda_c_quantile %nda_c_quantile% ^
    --eval_mode --comp_FID --FID_radius 0 --nfake_per_label 1000 ^
    --dump_fake_for_NIQE --niqe_dump_path %dump_niqe_path% ^ %*