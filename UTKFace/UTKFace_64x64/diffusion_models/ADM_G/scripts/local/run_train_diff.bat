@echo off

set ROOT_PATH="C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/UTKFace/UTKFace_64x64/diffusion_models/ADM_G"
set DATA_PATH="C:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/UTKFace"
@REM set EVAL_PATH="C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/UTKFace/UTKFace_64x64/evaluation/eval_models"
  

@REM @REM Some configs refer to https://github.com/openai/improved-diffusion; Class-conditional ImageNet-64 model (270M parameters, trained for 250K iterations)
set MODEL_FLAGS=--image_size 64 --attention_resolutions 32,16,8 --num_channels 128 --num_head_channels 64 --num_heads 4 --num_res_blocks 3 --learn_sigma True --class_cond True --resblock_updown True --use_new_attention_order True --use_scale_shift_norm True
set DIFFUSION_FLAGS=--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False
set TRAIN_FLAGS=--lr 1e-5 --weight_decay 1e-3 --batch_size 64 --use_fp16 True --log_interval 500 --lr_anneal_steps 100000 --save_interval 5000

python image_train.py ^
    --setup_name Setup1 ^
    --root_dir %ROOT_PATH% --data_dir %DATA_PATH% ^
    %MODEL_FLAGS% ^
    %DIFFUSION_FLAGS% ^
    %TRAIN_FLAGS% ^ %*


    @REM --resume_checkpoint .\output\exp_Setup1\diffusion\model020000.pt ^