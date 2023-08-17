#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=dingx92@163.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=UK64_CC
#SBATCH --output=%x-%j.out


module load arch/avx512 StdEnv/2020
module load cuda/11.0
module load python/3.8.2
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements_cedar.req

ROOT_PATH="/scratch/dingx92/CcGAN_with_NDA/UTKFace/UTKFace_64x64_2/CcGAN"
DATA_PATH="/project/6000538/dingx92/datasets/UTKFace/"
EVAL_PATH="/scratch/dingx92/CcGAN_with_NDA/UTKFace/UTKFace_64x64_2/evaluation/eval_models"

SEED=2021
NUM_WORKERS=0
MIN_LABEL=1
MAX_LABEL=60
IMG_SIZE=64
MAX_N_IMG_PER_LABEL=2000
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=200


NITERS=40000
BATCH_SIZE_G=512
BATCH_SIZE_D=512
NUM_D_STEPS=1
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


resume_niter=0
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan $resume_niter --loss_type_gan $LOSS_TYPE --num_D_steps $NUM_D_STEPS \
    --save_niters_freq 5000 --visualize_freq 2000 \
    --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
    --num_grad_acc_d $NUM_ACC_D --num_grad_acc_g $NUM_ACC_G \
    --lr_g $LR_G --lr_d $LR_D --dim_gan $DIM_GAN --dim_embed $DIM_EMBED \
    --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA \
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout \
    --comp_FID --FID_radius 0 --nfake_per_label 1000 \
    2>&1 | tee output_${GAN_ARCH}_${NITERS}.txt
