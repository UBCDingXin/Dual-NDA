#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=2-00:00
#SBATCH --mail-user=dingx92@163.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=UK64_CfGe
#SBATCH --output=%x-%j.out

module load arch/avx512 StdEnv/2020
module load gcc/11.3.0 python/3.11.2 cuda/11.8.0 opencv/4.8.0
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements_cedar.req

ROOT_PREFIX="/scratch/dingx92/CcGAN_with_NDA/UTKFace/UTKFace_64x64"
ROOT_PATH="${ROOT_PREFIX}/diffusion_models/classifier-free_guidance"
DATA_PATH="/project/6000538/dingx92/datasets/UTKFace/"
EVAL_PATH="${ROOT_PREFIX}/evaluation/eval_models"

SEED=2023
NUM_WORKERS=8
MIN_LABEL=1
MAX_LABEL=60
IMG_SIZE=64
MAX_N_IMG_PER_LABEL=99999
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=200

BATCH_SIZE=1024
GRAD_ACC=1
LR=1e-4
TIMESTEPS=1000
SAMP_TIMESTEPS=100

SETUP="Setup1"
NITERS=100000
RESUME_NITER=100000
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS --setting_name $SETUP \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --niters $NITERS --resume_niter $RESUME_NITER --timesteps $TIMESTEPS --sampling_timesteps $SAMP_TIMESTEPS \
    --train_batch_size $BATCH_SIZE --train_lr $LR --gradient_accumulate_every $GRAD_ACC \
    --train_amp \
    --sample_every 2000 --save_every 10000 \
    --comp_FID --nfake_per_label 1000 --samp_batch_size 1000 \
    --dump_fake_data \
    2>&1 | tee output_eval_${SETUP}_${NITERS}.txt