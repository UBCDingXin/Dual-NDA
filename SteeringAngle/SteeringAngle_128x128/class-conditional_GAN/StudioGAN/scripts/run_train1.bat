::===============================================================
:: This is a batch script for running the program on windows! 
:: Please set the following path correctly! 
:: Recommend using absolute path!
::===============================================================

@echo off

set ROOT_PREFIX=<Your_Path>/Dual-NDA/SteeringAngle/SteeringAngle_128x128
set DATA_PATH=<Your_Path>/Dual-NDA/datasets/SteeringAngle

set GAN_NAME=ReACGAN
set CONFIG_PATH=./configs/%GAN_NAME%.yaml

wandb disabled

python main.py --train -metrics none ^
    --data_dir %DATA_PATH% --cfg_file %CONFIG_PATH% --save_dir ./output/%GAN_NAME% ^
    --seed 2023 --num_workers 8 ^
    -sync_bn -mpc ^
    --print_freq 1000 --save_freq 5000 ^ %*