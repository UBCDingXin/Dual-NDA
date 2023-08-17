#!/bin/bash
# Please set the following path correctly! 
# Recommend using absolute path!

METHOD_NAME="CcGAN"
ROOT_PREFIX="<Your_Path>/Dual-NDA/UTKFace/UTKFace_64x64"
ROOT_PATH="${ROOT_PREFIX}/${METHOD_NAME}/baseline"
DATA_PATH="<Your_Path>/Dual-NDA/datasets/UTKFace"
EVAL_PATH="${ROOT_PREFIX}/evaluation/eval_models"
dump_niqe_path="<Your_Path>/Dual-NDA/NIQE/UTKFace/NIQE_filter_64x64/fake_data"


SEED=2021
NUM_WORKERS=0
MIN_LABEL=1
MAX_LABEL=60
IMG_SIZE=64
MAX_N_IMG_PER_LABEL=99999
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=200

BATCH_SIZE_G=256
BATCH_SIZE_D=256
NUM_D_STEPS=2
SIGMA=-1.0
KAPPA=-1.0
LR_G=1e-4
LR_D=1e-4
NUM_ACC_D=1
NUM_ACC_G=1

GAN_ARCH="SNGAN"
LOSS_TYPE="vanilla"
DIM_GAN=256
DIM_EMBED=128

NITERS=40000
SETTING="niters${NITERS}"
resume_niter=0
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan $resume_niter --loss_type_gan $LOSS_TYPE --num_D_steps $NUM_D_STEPS \
    --save_niters_freq 5000 --visualize_freq 2500 \
    --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
    --num_grad_acc_d $NUM_ACC_D --num_grad_acc_g $NUM_ACC_G \
    --lr_g $LR_G --lr_d $LR_D --dim_gan $DIM_GAN --dim_embed $DIM_EMBED \
    --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA \
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout \
    --setting_name $SETTING --samp_batch_size 2000 \
    --niqe_filter \
    --niqe_dump_path $dump_niqe_path --niqe_nfake_per_label_burnin 10000 \

