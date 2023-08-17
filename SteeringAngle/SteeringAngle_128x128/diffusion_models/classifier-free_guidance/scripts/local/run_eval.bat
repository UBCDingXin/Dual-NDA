@echo off

set ROOT_PREFIX=C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/SteeringAngle/SteeringAngle_128x128
set ROOT_PATH=%ROOT_PREFIX%/diffusion_models/classifier-free_guidance
set DATA_PATH=C:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/SteeringAngle
set EVAL_PATH=%ROOT_PREFIX%/evaluation/eval_models

set SEED=2023
set MIN_LABEL=-80.0
set MAX_LABEL=80.0
set IMG_SIZE=128
set MAX_N_IMG_PER_LABEL=9999
set MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0
set N_CLS=221

set BATCH_SIZE=128
set GRAD_ACC=1
set LR=1e-4
set TIMESTEPS=1000
set SAMP_TIMESTEPS=100

set SETUP="Setup1"
set NITERS=50000
set RESUME_NITER=0

set dump_niqe_path=F:/LocalWD/CcGAN_TPAMI_NIQE/SteeringAngle/NIQE_128x128/fake_data

python main.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% --setting_name %SETUP% ^
    --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% --num_classes %N_CLS% ^
    --max_num_img_per_label %MAX_N_IMG_PER_LABEL% --max_num_img_per_label_after_replica %MAX_N_IMG_PER_LABEL_AFTER_REPLICA% ^
    --niters %NITERS% --resume_niter %RESUME_NITER% --timesteps %TIMESTEPS% --sampling_timesteps %SAMP_TIMESTEPS% ^
    --train_batch_size %BATCH_SIZE% --train_lr %LR% --train_amp --gradient_accumulate_every %GRAD_ACC% ^
    --sample_every 2000 --save_every 5000 ^
    --comp_FID --nfake_per_label 50 --samp_batch_size 50 ^
    --dump_fake_data --dump_fake_for_NIQE --niqe_dump_path %dump_niqe_path% ^ %*