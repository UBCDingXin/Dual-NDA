#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=dingx92@163.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=SA64_SG2
#SBATCH --output=%x-%j.out


module load arch/avx512 StdEnv/2020
module load gcc/9.3.0 python/3.11.2 cuda/11.8.0 opencv/4.8.0
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements_cedar.req



ROOT_PREFIX="/scratch/dingx92/CcGAN_with_NDA/SteeringAngle/SteeringAngle_64x64"
DATA_PATH="/project/6000538/dingx92/datasets/SteeringAngle"
EVAL_PATH="${ROOT_PREFIX}/evaluation/eval_models"
CKPT_EVAL_FID_PATH="${EVAL_PATH}/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth"
CKPT_EVAL_LS_PATH="${EVAL_PATH}/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth"
CKPT_EVAL_Div_PATH="${EVAL_PATH}/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_20_seed_2020_classify_5_scenes_CVMode_False.pth"

GAN_NAME="ADCGAN"
CONFIG_PATH="./configs/${GAN_NAME}.yaml"

wandb disabled

python main.py --train -metrics none \
    --data_dir $DATA_PATH --cfg_file $CONFIG_PATH --save_dir "./output/${GAN_NAME}" \
    --seed 2023 --num_workers 2 \
    --print_freq 100 --save_freq 5000 \
    2>&1 | tee output_${GAN_NAME}.txt