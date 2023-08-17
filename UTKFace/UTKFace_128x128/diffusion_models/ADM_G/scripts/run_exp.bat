::===============================================================
:: This is a batch script for running the program on windows! 
:: Please set the following path correctly! 
:: Recommend using absolute path!
::===============================================================

@echo off

set ROOT_PATH=<Your_Path>/Dual-NDA/UTKFace/UTKFace_128x128/diffusion_models/ADM_G
set DATA_PATH=<Your_Path>/Dual-NDA/datasets/UTKFace
set EVAL_PATH=<Your_Path>/Dual-NDA/UTKFace/UTKFace_128x128/evaluation/eval_models
set dump_niqe_path=<Your_Path>/Dual-NDA/NIQE/UTKFace/NIQE_128x128/fake_data

set SETTING="Setup1"

::===============================================================
:: Train a classifier
::===============================================================
set CLASSIFIER_FLAGS=--classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_use_fp16 True

python classifier_train.py ^
    --setup_name %SETTING% ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% --image_size 128 ^
    --iterations 20000 --save_interval 5000 ^
    --batch_size 64 --lr 3e-4 --anneal_lr True --weight_decay 0.05 ^
    --log_interval 1000 ^
    %CLASSIFIER_FLAGS% ^ %*




::===============================================================
:: Train a diffusion model
::===============================================================
@REM @REM Some configs refer to https://github.com/openai/improved-diffusion; Class-conditional ImageNet-64 model (270M parameters, trained for 250K iterations)
set MODEL_FLAGS=--image_size 128 --attention_resolutions 32,16,8 --num_channels 128 --num_head_channels 64 --num_heads 4 --num_res_blocks 3 --learn_sigma True --class_cond True --resblock_updown True --use_new_attention_order True --use_scale_shift_norm True
set DIFFUSION_FLAGS=--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False
set TRAIN_FLAGS=--lr 1e-5 --weight_decay 1e-3 --batch_size 24 --use_fp16 True --log_interval 500 --lr_anneal_steps 50000 --save_interval 5000

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
set MODEL_FLAGS=--attention_resolutions 32,16,8 --num_channels 128 --num_head_channels 64 --num_heads 4 --num_res_blocks 3 --learn_sigma True --class_cond True --resblock_updown True --use_new_attention_order True --use_scale_shift_norm True --use_fp16 True
set SAMPLE_FLAGS=--nfake_per_label 1000 --samp_batch_size 50 --timestep_respacing ddim25 --use_ddim True
set EVAL_CONFIG=--eval_ckpt_path %EVAL_PATH% --dump_fake_data True --comp_FID True --dump_fake_for_NIQE True --niqe_dump_path %dump_niqe_path%

python classifier_sample.py ^
    --setup_name %SETTING% ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% --image_size 128 ^
    --model_path %path_to_model% ^
    --classifier_path %path_to_classifier% ^
    %CLASSIFIER_FLAGS% ^
    %MODEL_FLAGS% ^
    %SAMPLE_FLAGS% ^
    %EVAL_CONFIG% ^ %*