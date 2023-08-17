@echo off

set ROOT_PATH="C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/UTKFace/UTKFace_192x192/CcGAN"
set DATA_PATH="C:/BaiduSyncdisk/Baidu_WD/datasets/CCGM_or_regression/UTKFace"
set EVAL_PATH="C:/BaiduSyncdisk/Baidu_WD/CCGM/CcGAN_with_NDA/UTKFace/UTKFace_192x192/evaluation/eval_models"

@REM set CUDA_VISIBLE_DEVICES=1,0
python baseline_cnn.py ^
    --root_path %ROOT_PATH% --data_path %DATA_PATH% ^
    --cnn_name "vgg11" --img_size 192 ^ %*