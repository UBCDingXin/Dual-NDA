::===============================================================
:: This is a batch script for running the program on windows! 
:: Please set the following path correctly! 
:: Recommend using absolute path!
::===============================================================

@echo off

set ROOT_PATH=<Your_Path>/Dual-NDA/SteeringAngle/SteeringAngle_64x64/diffusion_models/ADM_G
set DATA_PATH=<Your_Path>/Dual-NDA/datasets/SteeringAngle
set EVAL_PATH=<Your_Path>/Dual-NDA/SteeringAngle/SteeringAngle_64x64/evaluation/eval_models
set dump_niqe_path=<Your_Path>/Dual-NDA/NIQE/SteeringAngle/NIQE_64x64/fake_data

set SETTING="Setup1"

::===============================================================
:: Train a classifier
::===============================================================
set CLASSIFIER_FLAGS=--classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_use_fp16 True

python classifier_train.py ^
    --setup_name %SETTING% ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% --image_size 64 ^
    --iterations 20000 --save_interval 5000 ^
    --batch_size 128 --lr 3e-4 --anneal_lr True --weight_decay 0.05 ^
    --log_interval 500 ^
    %CLASSIFIER_FLAGS% ^ %*




::===============================================================
:: Train a diffusion model
::===============================================================
@REM @REM Refer to https://github.com/openai/improved-diffusion; Class-conditional ImageNet-64 model (270M parameters, trained for 250K iterations)
set MODEL_FLAGS=--image_size 64 --num_channels 192 --num_res_blocks 3 --learn_sigma True --class_cond True
set DIFFUSION_FLAGS=--diffusion_steps 4000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False
set TRAIN_FLAGS=--lr 3e-4 --batch_size 32 --use_fp16 True --log_interval 500 --lr_anneal_steps 50000 --save_interval 5000

python image_train.py ^
    --setup_name %SETTING% ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% ^
    %MODEL_FLAGS% ^
    %DIFFUSION_FLAGS% ^
    %TRAIN_FLAGS% ^ %*




::===============================================================
:: Sample from the diffusion model guided by the classifier
::===============================================================
set path_to_model=%ROOT_PATH%/output/exp_%SETTING%/diffusion/model050000.pt
set path_to_classifier=%ROOT_PATH%/output/exp_%SETTING%/classifier/model019999.pt

set CLASSIFIER_FLAGS=--classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 True
set MODEL_FLAGS=--num_channels 192 --num_res_blocks 3 --learn_sigma True --class_cond True --use_fp16 True
set SAMPLE_FLAGS=--nfake_per_label 50 --samp_batch_size 200 --timestep_respacing ddim25 --use_ddim True
set EVAL_CONFIG=--eval_ckpt_path %EVAL_PATH% --dump_fake_data True --comp_FID True --niqe_dump_path %dump_niqe_path% --dump_fake_for_NIQE True

python classifier_sample.py ^
    --setup_name %SETTING% ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% --image_size 64 ^
    --model_path %path_to_model% ^
    --classifier_path %path_to_classifier% ^
    %CLASSIFIER_FLAGS% ^
    %MODEL_FLAGS% ^
    %SAMPLE_FLAGS% ^
    %EVAL_CONFIG% ^ %*