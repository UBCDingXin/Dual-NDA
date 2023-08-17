@echo off

set ROOT_PATH="C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/UTKFace/UTKFace_64x64/diffusion_models/ADM_G"
set DATA_PATH="C:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/UTKFace"
@REM set EVAL_PATH="C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/UTKFace/UTKFace_64x64/evaluation/eval_models"
  

set SETTING="Setup2"

@REM @REM Some configs refer to https://github.com/openai/improved-diffusion; Class-conditional ImageNet-64 model (270M parameters, trained for 250K iterations)
@REM Decay lr from 3e-4 to 1e-5 to encourage convergence
set MODEL_FLAGS=--image_size 64 --num_channels 192 --num_res_blocks 3 --learn_sigma True --class_cond True
set DIFFUSION_FLAGS=--diffusion_steps 4000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False
set TRAIN_FLAGS=--lr 1e-5 --batch_size 32 --use_fp16 True --log_interval 500 --lr_anneal_steps 50000 --save_interval 5000

@REM set CUDA_VISIBLE_DEVICES=0
python image_train.py ^
    --setup_name %SETTING% ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% ^
    %MODEL_FLAGS% ^
    %DIFFUSION_FLAGS% ^
    %TRAIN_FLAGS% ^ %*