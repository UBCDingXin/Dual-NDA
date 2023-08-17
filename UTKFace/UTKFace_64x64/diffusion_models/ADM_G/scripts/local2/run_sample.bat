@echo off

set ROOT_PATH="C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/UTKFace/UTKFace_64x64/diffusion_models/ADM_G"
set DATA_PATH="C:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/UTKFace"
set EVAL_PATH="C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/UTKFace/UTKFace_64x64/evaluation/eval_models"


set setup_name=Setup2
set path_to_model=%ROOT_PATH%/output/exp_%setup_name%/diffusion/model020000.pt
set path_to_classifier=%ROOT_PATH%/output/exp_%setup_name%/classifier/model019999.pt

set dump_niqe_path=F:/LocalWD/CcGAN_TPAMI_NIQE/UTKFace/NIQE_64x64/fake_data

set CLASSIFIER_FLAGS=--classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --classifier_use_fp16 True
set MODEL_FLAGS=--num_channels 192 --num_res_blocks 3 --learn_sigma True --class_cond True --use_fp16 True
set SAMPLE_FLAGS=--nfake_per_label 1000 --samp_batch_size 200 --timestep_respacing ddim25 --use_ddim True
set EVAL_CONFIG=--eval_ckpt_path %EVAL_PATH% --dump_fake_data True --comp_FID True --niqe_dump_path %dump_niqe_path%

set CUDA_VISIBLE_DEVICES=0
python classifier_sample.py ^
    --setup_name %setup_name% ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% --image_size 64 ^
    --model_path %path_to_model% ^
    --classifier_path %path_to_classifier% ^
    %CLASSIFIER_FLAGS% ^
    %MODEL_FLAGS% ^
    %SAMPLE_FLAGS% ^
    %EVAL_CONFIG% --dump_fake_for_NIQE True ^ %*