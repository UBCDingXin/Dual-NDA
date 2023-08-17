@echo off

set ROOT_PREFIX=C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/UTKFace/UTKFace_64x64
set ROOT_PATH=%ROOT_PREFIX%/diffusion_models/classifier-free_guidance
set DATA_PATH=C:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/UTKFace
set EVAL_PATH=%ROOT_PREFIX%/evaluation/eval_models

set SEED=2023
set NUM_WORKERS=0
set MIN_LABEL=1
set MAX_LABEL=60
set IMG_SIZE=64
set MAX_N_IMG_PER_LABEL=99999
set MAX_N_IMG_PER_LABEL_AFTER_REPLICA=200

set BATCH_SIZE=128
set GRAD_ACC=1
set LR=1e-4
set TIMESTEPS=1000
set SAMP_TIMESTEPS=100

set SETUP="Setup0"
set NITERS=50000
set RESUME_NITER=50000

set dump_niqe_path=F:/LocalWD/CcGAN_TPAMI_NIQE/UTKFace/NIQE_64x64/fake_data

python main.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% --eval_ckpt_path %EVAL_PATH% --seed %SEED% --num_workers %NUM_WORKERS% --setting_name %SETUP% ^
    --min_label %MIN_LABEL% --max_label %MAX_LABEL% --img_size %IMG_SIZE% ^
    --max_num_img_per_label %MAX_N_IMG_PER_LABEL% --max_num_img_per_label_after_replica %MAX_N_IMG_PER_LABEL_AFTER_REPLICA% ^
    --niters %NITERS% --resume_niter %RESUME_NITER% --timesteps %TIMESTEPS% --sampling_timesteps %SAMP_TIMESTEPS% ^
    --train_batch_size %BATCH_SIZE% --train_lr %LR% --gradient_accumulate_every %GRAD_ACC% ^
    --train_amp ^
    --sample_every 2000 --save_every 5000 ^
    --comp_FID --nfake_per_label 1000 --samp_batch_size 1000 ^
    --dump_fake_data --dump_fake_for_NIQE --niqe_dump_path %dump_niqe_path% ^ %*