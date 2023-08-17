#!/bin/bash
# Please set the following path correctly! 
# Recommend using absolute path!

METHOD_NAME="CcGAN"
ROOT_PREFIX="<Your_Path>/Dual-NDA/UTKFace/UTKFace_128x128"
ROOT_PATH="${ROOT_PREFIX}/${METHOD_NAME}/NDA"
DATA_PATH="<Your_Path>/Dual-NDA/datasets/UTKFace"
EVAL_PATH="${ROOT_PREFIX}/evaluation/eval_models"
dump_niqe_path="<Your_Path>/Dual-NDA/NIQE/UTKFace/NIQE_128x128/fake_data"

SEED=2023
NUM_WORKERS=0
MIN_LABEL=1
MAX_LABEL=60
IMG_SIZE=128
MAX_N_IMG_PER_LABEL=2000
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=200

BATCH_SIZE_G=128
BATCH_SIZE_D=128
NUM_D_STEPS=4
SIGMA=-1.0
KAPPA=-2.0
LR_G=1e-4
LR_D=1e-4
NUM_ACC_D=2
NUM_ACC_G=2

GAN_ARCH="SAGAN"
LOSS_TYPE="hinge"
DIM_GAN=256
DIM_EMBED=128

SETTING="Setup_vNDA"

fake_data_path_1="${ROOT_PREFIX}/${METHOD_NAME}/baseline/output/SAGAN_soft_si0.041_ka900.000_hinge_nDs4_nDa1_nGa1_Dbs256_Gbs256/bad_fake_data/niters20K/badfake_NIQE0.9_nfake60000.h5"
nda_c_quantile=0.9
nda_start_iter=0

NITERS=20000
resume_niter=0
python main.py \
    --setting_name $SETTING \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan $resume_niter --loss_type_gan $LOSS_TYPE --num_D_steps $NUM_D_STEPS \
    --save_niters_freq 2500 --visualize_freq 2000 \
    --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
    --num_grad_acc_d $NUM_ACC_D --num_grad_acc_g $NUM_ACC_G \
    --lr_g $LR_G --lr_d $LR_D --dim_gan $DIM_GAN --dim_embed $DIM_EMBED \
    --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA \
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout \
    --nda_start_iter $nda_start_iter \
    --nda_a 0.25 --nda_b 0.75 --nda_c 0 --nda_d 0 --nda_e 0 --nda_c_quantile $nda_c_quantile \
    --nda_d_nfake $nfake_d \
    --path2badfake1 $fake_data_path_1 \
    --comp_FID --FID_radius 0 --nfake_per_label 1000 \
    --dump_fake_for_NIQE --dump_fake_img_path $dump_niqe_path \