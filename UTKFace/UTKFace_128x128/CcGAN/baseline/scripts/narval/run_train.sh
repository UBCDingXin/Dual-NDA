#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=dingx92@163.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=UK128_BL
#SBATCH --output=%x-%j.out


module load StdEnv/2020 
module load cuda/11.4 python/3.9 opencv/4.5.5
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements_narval.req

METHOD_NAME="CcGAN_v3"
ROOT_PREFIX="/scratch/dingx92/CcGAN_with_NDA/UTKFace/UTKFace_128x128"
ROOT_PATH="${ROOT_PREFIX}/${METHOD_NAME}/baseline"
DATA_PATH="/project/6000538/dingx92/datasets/UTKFace/"
EVAL_PATH="${ROOT_PREFIX}/evaluation/eval_models"

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

NITERS=20000
resume_niter=0
SETTING="niters20K"
python main.py \
    --setting_name $SETTING \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan $resume_niter --loss_type_gan $LOSS_TYPE --num_D_steps $NUM_D_STEPS \
    --save_niters_freq 2500 --visualize_freq 1000 \
    --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
    --num_grad_acc_d $NUM_ACC_D --num_grad_acc_g $NUM_ACC_G \
    --lr_g $LR_G --lr_d $LR_D --dim_gan $DIM_GAN --dim_embed $DIM_EMBED \
    --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA \
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout \
    --comp_FID --FID_radius 0 --nfake_per_label 1000 \
    2>&1 | tee output_${GAN_ARCH}_${NITERS}_${SETTING}.txt