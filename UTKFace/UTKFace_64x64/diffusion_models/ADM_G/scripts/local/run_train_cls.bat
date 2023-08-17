@echo off

set ROOT_PATH="C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/UTKFace/UTKFace_64x64/diffusion_models/ADM_G"
set DATA_PATH="C:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/UTKFace"
@REM set EVAL_PATH="C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/UTKFace/UTKFace_64x64/evaluation/eval_models"


@REM python classifier_train.py ^
@REM     --setup_name Setup1 ^
@REM     --root_dir %ROOT_PATH% --data_dir %DATA_PATH% --image_size 64 ^
@REM     --iterations 100 --save_interval 100 ^
@REM     --batch_size 128 --lr 3e-4 --anneal_lr True --weight_decay 0.05 ^
@REM     --classifier_attention_resolutions 32,16,8 --classifier_depth 2 ^
@REM     --classifier_width 128 --classifier_pool attention ^
@REM     --classifier_resblock_updown True --classifier_use_scale_shift_norm True ^
@REM     --classifier_use_fp16 True ^
@REM     --log_interval 100 ^ %*

set CLASSIFIER_FLAGS=--classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_use_fp16 True

python classifier_train.py ^
    --setup_name Setup1 ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% --image_size 64 ^
    --iterations 100 --save_interval 100 ^
    --batch_size 128 --lr 3e-4 --anneal_lr True --weight_decay 0.05 ^
    --log_interval 100 ^
    %CLASSIFIER_FLAGS% ^ %*



@REM MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
@REM python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/64x64_classifier.pt --classifier_depth 4 --model_path models/64x64_diffusion.pt $SAMPLE_FLAGS