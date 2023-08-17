#!/bin/bash
# Please set the following path correctly! 
# Recommend using absolute path!

METHOD_NAME=CcGAN
ROOT_PREFIX="<Your_Path>/Dual-NDA/SteeringAngle/SteeringAngle_128x128"
ROOT_PATH="${ROOT_PREFIX}/${METHOD_NAME}/NDA"
DATA_PATH="<Your_Path>/Dual-NDA/datasets/SteeringAngle"
EVAL_PATH="${ROOT_PREFIX}/evaluation/eval_models"
niqe_dump_path="<Your_Path>/Dual-NDA/NIQE/SteeringAngle/NIQE_128x128/fake_data"

SEED=2023
NUM_WORKERS=0
MIN_LABEL=-80.0
MAX_LABEL=80.0
IMG_SIZE=128
MAX_N_IMG_PER_LABEL=9999
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0

BATCH_SIZE_G=128
BATCH_SIZE_D=128
NUM_D_STEPS=2
SIGMA=-1.0
KAPPA=-5.0
LR_G=1e-4
LR_D=1e-4
NUM_ACC_D=2
NUM_ACC_G=2

GAN_ARCH="SAGAN"
LOSS_TYPE="hinge"
DIM_GAN=256
DIM_EMBED=128


SETTING="Setup1"

fake_data_path_1="${ROOT_PREFIX}/${METHOD_NAME}/baseline/output/SAGAN_soft_si0.029_ka1000.438_hinge_nDs2_nDa1_nGa1_Dbs256_Gbs256/bad_fake_data/niters20K/badfake_NIQE0.9_nfake17740.h5"
nda_c_quantile=0.5
nda_start_iter=15000

NITERS=20000
resume_niter=0
python main.py \
    --setting_name $SETTING --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan $resume_niter --loss_type_gan $LOSS_TYPE --num_D_steps $NUM_D_STEPS \
    --save_niters_freq 2500 --visualize_freq 500 \
    --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
    --num_grad_acc_d $NUM_ACC_D --num_grad_acc_g $NUM_ACC_G \
    --lr_g $LR_G --lr_d $LR_D --dim_gan $DIM_GAN --dim_embed $DIM_EMBED \
    --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA \
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout \
    --nda_start_iter $nda_start_iter \
    --nda_a 0.5 --nda_b 0 --nda_c 0.2 --nda_d 0.3 --nda_e 0 --nda_c_quantile $nda_c_quantile \
    --path2badfake1 $fake_data_path_1 \
    --comp_FID --samp_batch_size 200 --dump_fake_for_NIQE --niqe_dump_path $niqe_dump_path \