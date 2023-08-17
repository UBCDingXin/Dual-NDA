#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=0-12:00
#SBATCH --mail-user=dingx92@163.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=SA64_BL
#SBATCH --output=%x-%j.out


module load StdEnv/2020
module load cuda/11.4 python/3.9 opencv/4.5.5
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements_narval.req


METHOD_NAME="CcGAN"
ROOT_PATH="/scratch/dingx92/CcGAN_with_NDA/SteeringAngle/SteeringAngle_64x64/${METHOD_NAME}/baseline"
DATA_PATH="/project/6000538/dingx92/datasets/SteeringAngle/"
EVAL_PATH="/scratch/dingx92/CcGAN_with_NDA/SteeringAngle/SteeringAngle_64x64/evaluation/eval_models"

SEED=2023
NUM_WORKERS=0
MIN_LABEL=-80.0
MAX_LABEL=80.0
IMG_SIZE=64
MAX_N_IMG_PER_LABEL=9999
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0

BATCH_SIZE_G=512
BATCH_SIZE_D=512
NUM_D_STEPS=2
SIGMA=-1.0
KAPPA=-5.0
LR_G=1e-4
LR_D=1e-4
NUM_ACC_D=1
NUM_ACC_G=1

GAN_ARCH="SAGAN"
LOSS_TYPE="hinge"

DIM_GAN=256
DIM_EMBED=128

NITERS=20000
Setting="niters20K"
resume_niter=0
python main.py \
    --setting_name $Setting \
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
    --comp_FID \
    2>&1 | tee output_BL1_${GAN_ARCH}_${NITERS}.txt
