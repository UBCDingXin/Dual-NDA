#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=0-12:00
#SBATCH --mail-user=dingx92@163.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=UK64
#SBATCH --output=%x-%j.out


module load StdEnv/2020 gcc/9.3.0 cuda/11.4 python/3.9 opencv/4.5.5
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements_narval.req

ROOT_PATH="/scratch/dingx92/CcGAN_with_NDA/UTKFace/UTKFace_64x64/CcGAN"
DATA_PATH="/project/6000538/dingx92/datasets/UTKFace/"
EVAL_PATH="/scratch/dingx92/CcGAN_with_NDA/UTKFace/UTKFace_64x64/evaluation/eval_models"

python baseline_cnn.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH \
    --cnn_name "vgg11" --img_size 64 \
    2>&1 | tee output_vgg11.txt