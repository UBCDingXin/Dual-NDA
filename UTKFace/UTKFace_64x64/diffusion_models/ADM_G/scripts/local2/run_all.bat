@echo off

set ROOT_PATH="C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/UTKFace/UTKFace_64x64/diffusion_models/ADM_G"
set DATA_PATH="C:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/UTKFace"
set EVAL_PATH="C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/UTKFace/UTKFace_64x64/evaluation/eval_models"

set SETTING="Setup2"
set CUDA_VISIBLE_DEVICES=0



@REM set CLASSIFIER_FLAGS=--classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_use_fp16 True

@REM python classifier_train.py ^
@REM     --setup_name %SETTING% ^
@REM     --root_dir %ROOT_PATH% --data_dir %DATA_PATH% --image_size 64 ^
@REM     --iterations 20000 --save_interval 5000 ^
@REM     --batch_size 128 --lr 3e-4 --anneal_lr True --weight_decay 0.05 ^
@REM     --log_interval 500 ^
@REM     %CLASSIFIER_FLAGS% ^ %*




@REM @REM Some configs refer to https://github.com/openai/improved-diffusion; Class-conditional ImageNet-64 model (270M parameters, trained for 250K iterations)
@REM Decay lr from 3e-4 to 1e-5 to encourage convergence
set MODEL_FLAGS=--image_size 64 --num_channels 192 --num_res_blocks 3 --learn_sigma True --class_cond True
set DIFFUSION_FLAGS=--diffusion_steps 4000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False
set TRAIN_FLAGS=--lr 1e-5 --batch_size 32 --use_fp16 True --log_interval 500 --lr_anneal_steps 50000 --save_interval 5000

python image_train.py ^
    --setup_name %SETTING% ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% ^
    %MODEL_FLAGS% ^
    %DIFFUSION_FLAGS% ^
    %TRAIN_FLAGS% ^ %*





set path_to_model=%ROOT_PATH%/output/exp_%SETTING%/diffusion/model050000.pt
set path_to_classifier=%ROOT_PATH%/output/exp_%SETTING%/classifier/model019999.pt
set dump_niqe_path=F:/LocalWD/CcGAN_TPAMI_NIQE/UTKFace/NIQE_64x64/fake_data

set CLASSIFIER_FLAGS=--classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 True
set MODEL_FLAGS=--num_channels 192 --num_res_blocks 3 --learn_sigma True --class_cond True --use_fp16 True
set SAMPLE_FLAGS=--nfake_per_label 1000 --samp_batch_size 200 --timestep_respacing ddim25 --use_ddim True
set EVAL_CONFIG=--eval_ckpt_path %EVAL_PATH% --dump_fake_data True --comp_FID True
@REM --dump_fake_for_NIQE True --niqe_dump_path %dump_niqe_path%

python classifier_sample.py ^
    --setup_name %SETTING% ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% --image_size 64 ^
    --model_path %path_to_model% ^
    --classifier_path %path_to_classifier% ^
    %CLASSIFIER_FLAGS% ^
    %MODEL_FLAGS% ^
    %SAMPLE_FLAGS% ^
    %EVAL_CONFIG% ^ %*
