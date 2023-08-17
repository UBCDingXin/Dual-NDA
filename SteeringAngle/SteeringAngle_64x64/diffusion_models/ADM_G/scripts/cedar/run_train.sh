#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --mail-user=dingx92@163.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=UK64_ADM1
#SBATCH --output=%x-%j.out

module load arch/avx512 StdEnv/2020
module load gcc/9.3.0 python/3.11.2 cuda/11.8.0 opencv/4.8.0 openmpi/4.0.3
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements_cedar.req


ROOT_PREFIX="/scratch/dingx92/CcGAN_with_NDA/UTKFace/UTKFace_64x64"
ROOT_PATH="${ROOT_PREFIX}/diffusion_models/ADM_G"
DATA_PATH="/project/6000538/dingx92/datasets/UTKFace/"
EVAL_PATH="${ROOT_PREFIX}/evaluation/eval_models"

SETTING="Setup1"


###################################################################################
## Classifier training

CLASSIFIER_FLAGS=--classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_use_fp16 True

python classifier_train.py \
    --setup_name $SETTING \
    --root_dir $ROOT_PATH --data_dir $DATA_PATH --image_size 64 \
    --iterations 20000 --save_interval 5000 \
    --batch_size 128 --lr 3e-4 --anneal_lr True --weight_decay 0.05 \
    --log_interval 500 \
    $CLASSIFIER_FLAGS \
    2>&1 | tee output_${SETTING}_cls_train.txt


###################################################################################
## Diffusion training

CONFIGS=--batch_size 128 --lr 3e-4 --lr_anneal_steps 50000 --weight_decay 1e-3 --attention_resolutions 32,16,8 --class_cond True --learn_sigma True --num_channels 64 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --log_interval 1000 --save_interval 5000

python image_train.py \
    --setup_name $SETTING \
    --root_dir $ROOT_PATH --data_dir $DATA_PATH --image_size 64 \
    $CONFIGS \
    2>&1 | tee output_${SETTING}_diff_train.txt


###################################################################################
## Sampling and evaluation
