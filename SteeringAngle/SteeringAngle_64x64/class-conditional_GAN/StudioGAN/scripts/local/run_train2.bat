::===============================================================
:: This is a batch script for running the program on windows! 
::===============================================================

@echo off

set ROOT_PREFIX=C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/SteeringAngle/SteeringAngle_64x64
set DATA_PATH=C:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/SteeringAngle
set EVAL_PATH=%ROOT_PREFIX%/evaluation/eval_models
set CKPT_EVAL_FID_PATH=%EVAL_PATH%/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth
set CKPT_EVAL_LS_PATH=%EVAL_PATH%/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth
set CKPT_EVAL_Div_PATH=%EVAL_PATH%/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_20_seed_2020_classify_5_scenes_CVMode_False.pth

set GAN_NAME=ADCGAN
set CONFIG_PATH=./configs/%GAN_NAME%.yaml

wandb disabled
@REM wandb online


python main.py --train -metrics none ^
    --data_dir %DATA_PATH% --cfg_file %CONFIG_PATH% --save_dir ./output/%GAN_NAME% ^
    --seed 2023 --num_workers 8 ^
    --print_freq 100 --save_freq 2000 ^ %*


@REM set CKPT_G=./output/%GAN_NAME%/checkpoints/SA64-ADCGAN-train-2023_08_04_06_20_54/model=G-current-weights-step=20000.pth
@REM set CKPT_G_EMA=./output/%GAN_NAME%/checkpoints/SA64-ADCGAN-train-2023_08_04_06_20_54/model=G_ema-current-weights-step=20000.pth
@REM set dump_niqe_path=F:/LocalWD/CcGAN_TPAMI_NIQE/SteeringAngle/NIQE_64x64/fake_data

@REM @REM set CUDA_VISIBLE_DEVICES=0,1
@REM python main.py -metrics none ^
@REM     --data_dir %DATA_PATH% --cfg_file %CONFIG_PATH% --save_dir ./output/%GAN_NAME% ^
@REM     --seed 2023 --num_workers 4 ^
@REM     --do_eval ^
@REM     --path_to_G %CKPT_G% --path_to_G_ema %CKPT_G_EMA% ^
@REM     --eval_ckpt_path_FID %CKPT_EVAL_FID_PATH% --eval_ckpt_path_LS %CKPT_EVAL_LS_PATH% --eval_ckpt_path_Div %CKPT_EVAL_Div_PATH% ^
@REM     --samp_batch_size 200 --eval_batch_size 200 ^
@REM     --dump_fake_for_NIQE --dump_fake_img_path %dump_niqe_path% ^ %*