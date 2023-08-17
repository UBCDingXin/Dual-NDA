#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100l:2
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=xin.ding@stat.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --job-name=CC_GEN
#SBATCH --output=%x-%j.out


module load arch/avx512 StdEnv/2020
module load cuda/11.0
module load python/3.8.2
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements_cedar.req

ROOT_PATH="/scratch/dingx92/CcGAN_with_NDA/UTKFace/UTKFace_192x192/CcGAN"
DATA_PATH="/project/6000538/dingx92/datasets/UTKFace/"
EVAL_PATH="/scratch/dingx92/CcGAN_with_NDA/UTKFace/UTKFace_192x192/evaluation/eval_models"

SEED=2023
NUM_WORKERS=0
MIN_LABEL=1
MAX_LABEL=60
IMG_SIZE=192
MAX_N_IMG_PER_LABEL=99999
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=200


NITERS=40000
BATCH_SIZE_G=120
BATCH_SIZE_D=120
NUM_D_STEPS=2
SIGMA=-1.0
KAPPA=-1.0
LR_G=1e-4
LR_D=1e-4
NUM_ACC_D=2
NUM_ACC_G=2

GAN_ARCH="SAGAN"
LOSS_TYPE="hinge"


DIM_GAN=256
DIM_EMBED=128


resume_niter=0
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan $resume_niter --loss_type_gan $LOSS_TYPE --num_D_steps $NUM_D_STEPS \
    --save_niters_freq 2500 --visualize_freq 2500 \
    --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
    --num_grad_acc_d $NUM_ACC_D --num_grad_acc_g $NUM_ACC_G \
    --lr_g $LR_G --lr_d $LR_D --dim_gan $DIM_GAN --dim_embed $DIM_EMBED \
    --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA \
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout \
    --comp_FID --FID_radius 0 --nfake_per_label 1000 \
    2>&1 | tee output_${GAN_ARCH}.txt


# resume_niter=0
# python main.py \
#     --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
#     --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
#     --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
#     --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan $resume_niter --loss_type_gan $LOSS_TYPE --num_D_steps $NUM_D_STEPS \
#     --save_niters_freq 2500 --visualize_freq 2500 \
#     --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
#     --num_grad_acc_d $NUM_ACC_D --num_grad_acc_g $NUM_ACC_G \
#     --lr_g $LR_G --lr_d $LR_D --dim_gan $DIM_GAN --dim_embed $DIM_EMBED \
#     --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA \
#     --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout \
#     --comp_FID --FID_radius 0 --nfake_per_label 1000 \
#     --gen_low_quality \
#     --samp_nfake_per_label_burnin 2000 --samp_nfake_per_label 20000 \
#     --samp_filter_precnn_net vgg11 --samp_filter_mae_percentile_threshold 0.85 \
#     --samp_filter_batch_size 500 --samp_batch_size 500 \
#     2>&1 | tee output_${GAN_ARCH}_gen.txt
