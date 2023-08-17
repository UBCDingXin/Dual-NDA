# **The Code Repository for Dual-NDA**


<!-- --------------------------------------------------------------- -->
## 1. Repository Structure

```
├── UTKFace
│   ├── UTKFace_64x64
│   │   ├──CcGAN
|   |   |  ├──baseline
|   |   |  └──NDA
│   │   ├──class-conditional_GAN
|   |   |  └──StudioGAN
|   |   ├──diffusion_models
|   |   |  ├──ADM_G
|   |   |  └──classifier_free_guidance
│   │   └──evaluation
|   |      └──eval_models
│   └── UTKFace_128x128
│       ├──CcGAN
|       |  ├──baseline
|       |  └──NDA
│       ├──class-conditional_GAN
|       |  └──StudioGAN
|       ├──diffusion_models
|       |  ├──ADM_G
|       |  └──classifier_free_guidance
│       └──evaluation
|          └──eval_models
├── SteeringAngle
│   ├── SteeringAngle_64x64
│   │   ├──CcGAN
|   |   |  ├──baseline
|   |   |  └──NDA
│   │   ├──class-conditional_GAN
|   |   |  └──StudioGAN
|   |   ├──diffusion_models
|   |   |  ├──ADM_G
|   |   |  └──classifier_free_guidance
│   │   └──evaluation
|   |      └──eval_models
│   └── SteeringAngle_128x128
│       ├──CcGAN
|       |  ├──baseline
|       |  └──NDA
│       ├──class-conditional_GAN
|       |  └──StudioGAN
|       ├──diffusion_models
|       |  ├──ADM_G
|       |  └──classifier_free_guidance
│       └──evaluation
|          └──eval_models
└── NIQE
    ├── UTKFace
    │   ├── NIQE_64x64
    │   ├── NIQE_128x128
    │   ├── NIQE_filter_64x64
    │   └── NIQE_filter_128x128
    └── SteeringAngle
        ├── NIQE_64x64
        ├── NIQE_128x128     
        ├── NIQE_filter_64x64
        └── NIQE_filter_128x128
```


<!-- --------------------------------------------------------------- -->
## 2. Software Requirements
Here, we provide a list of crucial software environments and python packages employed in the conducted experiments. Please note that we use different computational platforms for our experiments. <br />

**For computing NIQE scores and implementing the NIQE filtering:**
| Item | Version |
|---|---|
| OS | Win11 |
| Python | 3.11.3 |
| Matlab | 2023a |

**For training CcGAN:**
| Item | Version |
|---|---|
| OS | Linux |
| Python | 3.9 |
| CUDA  | 11.4 |
| numpy | 1.23.0 |
| torch | 1.12.1 |
| torchvision | 0.13.1 |
| Pillow | 8.4.0 |
| accelearate | 0.18.0 |
| matplotlib | 3.4.2 |

**For training ReACGAN, ADCGAN, ADM-G, and CFG:**
| Item | Version |
|---|---|
| OS | Win11 |
| Python | 3.11.3 |
| CUDA  | 11.8 |
| numpy | 1.23.5 |
| torch | 2.0.1 |
| torchvision | 0.15.2 |
| Pillow | 9.5.0 |
| accelearate | 0.20.3 |
| wandb | 0.15.7 |
| matplotlib | 3.7.1 |


<!-- --------------------------------------------------------------- -->
## 3. Datasets

We use the preprocessed datasets provided by [Ding et. al. (2023)](https://github.com/UBCDingXin/improved_CcGAN).

### The preprocessed UTKFace Dataset (h5 file)
Download the following h5 files and put them in `./datasets/UTKFace`.
#### UTKFace (64x64)
[UTKFace_64x64_download_link](https://1drv.ms/u/s!Arj2pETbYnWQstIzurW-LCFpGz5D7Q?e=X23ybx) <br />
#### UTKFace (128x128)
[UTKFace_128x128_download_link](https://1drv.ms/u/s!Arj2pETbYnWQstJGpTgNYrHE8DgDzA?e=d7AeZq) <br />

### The Steering Angle dataset (h5 file)
Download the following h5 files and put them in `./datasets/SteeringAngle`.
#### Steering Angle (64x64)
[SteeringAngle_64x64_download_link](https://1drv.ms/u/s!Arj2pETbYnWQstIyDTDpGA0CNiONkA?e=Ui5kUK) <br />
#### Steering Angle (128x128)
[SteeringAngle_128x128_download_link](https://1drv.ms/u/s!Arj2pETbYnWQstJ0j7rXhDtm6y4IcA?e=bLQh2e) <br />




<!-- --------------------------------------------------------------- -->
## 4. Training
As illustrated in the aforementioned repository structure, distinct training codes have been provided for various datasets. <br />


<!------------------------------------>
### (0) Download necessary checkpoints
#### Evaluation Checkpoints

For each experiment, download the corresponding zip file of checkpoints of the evaluation models. Then, unzip the zip file and copy the folder `eval_models` to `./XXX/XXX_YYY/evaluation`, where `XXX` is the dataset name (either `UTKFace` or `SteeringAngle`) and `YYY` represents the image resolution (either `64x64` or `128x128`)

[SteeringAngle_64x64_download_link](https://1drv.ms/u/s!Arj2pETbYnWQvMwHlZ362YyXnGuFXg?e=a0usQC) <br />
[SteeringAngle_128x128_download_link](https://1drv.ms/u/s!Arj2pETbYnWQvMwIx6X1bMIVTKAj5Q?e=SD9P0S) <br />
[UTKFace_64x64_download_link](https://1drv.ms/u/s!Arj2pETbYnWQvMtFOZQSCMFuojGBmg?e=JsdVks) <br />
[UTKFace_128x128_download_link](https://1drv.ms/u/s!Arj2pETbYnWQvMwM7hKjcN0IVVRMPg?e=cbmCtP) <br />

<!------------------------------------>
### (1) UTKFace (64x64)

<font color=Red>**!!! Please accurately configure the path parameters within each `.bat` or `.sh` file.!!!**</font>

* **Baseline CcGAN (SVDL+ILI)** <br />
Please go the the directory `./UTKFace/UTKFace_64x64/CcGAN/baseline`. Run the training script `./scripts/run_train.bat` for Windows or `./scripts/run_train.sh` for Linux.  We adopt the SNGAN network structure, the vanilla cGAN loss, the soft vicinity, and the improved label input mechanism. The models are trained for 40000 iterations, employing a batch size of 256. Specifically, the discriminator is updated twice for each iteration, while the generator is updated once. <br />
We also provide the checkpoint of the pre-trained CcGAN, which can be downloaded from [link](https://1drv.ms/u/s!Arj2pETbYnWQvMwOHj9m2OHRyvtetQ?e=jogJy7).

* **Dual-NDA** <br />
**First**, go to the directory `./UTKFace/UTKFace_64x64/CcGAN/baseline`. Run the data generation script `./scripts/run_gene.bat` for Windows or `./scripts/run_gene.sh` for Linux. During this generation, 10,000 fake samples will be created for each of the 60 age values. These samples will be stored in `./NIQE/UTKFace/NIQE_filter_64X64/fake_data/fake_images`. <br />
**Second**, go the directory `./NIQE/UTKFace/NIQE_filter_64X64`. Run the filtering scripts `./run_test1.bat` and `./run_test2.bat` sequentially. A `.h5` file with Type II negative samples will be generated to `./fake_data`. Move this `.h5` file to `./UTKFace/UTKFace_64x64/CcGAN/baseline/output/CcGAN_SNGAN_soft_si0.041_ka3600.000_vanilla_nDs2_nDa1_nGa1_Dbs256_Gbs256/bad_fake_data/niters40000`. <br />
**Third**, go to the directory `./UTKFace/UTKFace_64x64/CcGAN/NDA`. Run the training script `./scripts/run_dual.bat` for Windows or `./scripts/run_dual.sh` for Linux. During this process, we train CcGAN with Dual-NDA for 60000 iterations with the Dual-NDA mechanism being activated starting after the first 40000 iterations.

* **Vanilla NDA** <br />
Go to the directory `./UTKFace/UTKFace_64x64/CcGAN/NDA`. Run the training script `./scripts/run_nda.bat` for Windows or `./scripts/run_nda.sh` for Linux.

* **ReACGAN and ADCGAN** <br />
Go to the directory `./UTKFace/UTKFace_64x64/class-conditional_GAN/StudioGAN`. Run the training script `./scripts/run_train1.bat` for ReACGAN or `./scripts/run_train2.bat` for ADCGAN. We conduct the evaluation process by running `./scripts/run_eval1.bat` for ReACGAN or `./scripts/run_eval2.bat` for ADCGAN.

* **ADM-G (Classifier Guidance)** <br />
Go to `./UTKFace/UTKFace_64x64/diffusion_models/ADM_G`. Then, run the script `./scripts/run_exp.bat`. 
    
* **CFG (Classifier-Free Guidance)** <br />
Go to `./UTKFace/UTKFace_64x64/diffusion_models/classifier-free_guidance`. Then, run the script `./scripts/run_exp.bat`. 

<!------------------------------------>
### (2) UTKFace (128x128)

<font color=Red>**!!! Please accurately configure the path parameters within each `.bat` or `.sh` file.!!!**</font>

* **Baseline CcGAN (SVDL+ILI)** <br />
We use the checkpoint of a pre-trained CcGAN provided by Ding et. al. (2023), which can be downloaded from [link](https://1drv.ms/u/s!Arj2pETbYnWQvMwTiwLAlsW-rNy6Ww?e=98q2eH). Unzip it and put the folder we get in `./output`.

* **Dual-NDA** <br />
**First**, go to the directory `./UTKFace/UTKFace_128x128/CcGAN/baseline`. Run the data generation script `./scripts/run_gene.bat` for Windows or `./scripts/run_gene.sh` for Linux. During this generation, 10,000 fake samples will be created for each of the 60 age values. These samples will be stored in `./NIQE/UTKFace/NIQE_filter_128x128/fake_data/fake_images`. <br />
**Second**, go the directory `./NIQE/UTKFace/NIQE_filter_128x128`. Run the filtering scripts `./run_test1.bat` and `./run_test2.bat` sequentially. A `.h5` file with Type II negative samples will be generated to `./fake_data`. Move this `.h5` file to `./UTKFace/UTKFace_128x128/CcGAN/baseline/output/SAGAN_soft_si0.041_ka900.000_hinge_nDs4_nDa1_nGa1_Dbs256_Gbs256/bad_fake_data/niters20K`. <br />
**Third**, go to the directory `./UTKFace/UTKFace_128x128/CcGAN/NDA`. Run the training script `./scripts/run_dual.bat` for Windows or `./scripts/run_dual.sh` for Linux. During this process, we train CcGAN with Dual-NDA for 22500 iterations with the Dual-NDA mechanism being activated starting after the first 20000 iterations.

* **Vanilla NDA** <br />
Go to the directory `./UTKFace/UTKFace_128x128/CcGAN/NDA`. Run the training script `./scripts/run_nda.bat` for Windows or `./scripts/run_nda.sh` for Linux.

* **ReACGAN and ADCGAN** <br />
Go to the directory `./UTKFace/UTKFace_128x128/class-conditional_GAN/StudioGAN`. Run the training script `./scripts/run_train1.bat` for ReACGAN or `./scripts/run_train2.bat` for ADCGAN. We conduct the evaluation process by running `./scripts/run_eval1.bat` for ReACGAN or `./scripts/run_eval2.bat` for ADCGAN.

* **ADM-G (Classifier Guidance)** <br />
Go to `./UTKFace/UTKFace_128x128/diffusion_models/ADM_G`. Then, run the script `./scripts/run_exp.bat`. 
    
* **CFG (Classifier-Free Guidance)** <br />
Go to `./UTKFace/UTKFace_128x128/diffusion_models/classifier-free_guidance`. Then, run the script `./scripts/run_exp.bat`. 


<!------------------------------------>
### (3) Steering Angle (64x64)

<font color=Red>**!!! Please accurately configure the path parameters within each `.bat` or `.sh` file.!!!**</font>

![**!!! Please accurately configure the path parameters within each `.bat` or `.sh` file.!!!**](https://placehold.co/15x15/f03c15/f03c15.png) `#f03c15`

* **Baseline CcGAN (SVDL+ILI)** <br />
Please go the the directory `./UTKFace/UTKFace_64x64/CcGAN/baseline`. Run the training script `./scripts/run_train.bat` for Windows or `./scripts/run_train.sh` for Linux.  We adopt the SNGAN network structure, the vanilla cGAN loss, the soft vicinity, and the improved label input mechanism. The models are trained for 40000 iterations, employing a batch size of 256. Specifically, the discriminator is updated twice for each iteration, while the generator is updated once. <br />
We also provide the checkpoint of the pre-trained CcGAN, which can be downloaded from [link](https://1drv.ms/u/s!Arj2pETbYnWQvMwOHj9m2OHRyvtetQ?e=jogJy7).

* **Dual-NDA** <br />
**First**, go to the directory `./UTKFace/UTKFace_64x64/CcGAN/baseline`. Run the data generation script `./scripts/run_gene.bat` for Windows or `./scripts/run_gene.sh` for Linux. During this generation, 10,000 fake samples will be created for each of the 60 age values. These samples will be stored in `./NIQE/UTKFace/NIQE_filter_64X64/fake_data/fake_images`. <br />
**Second**, go the directory `./NIQE/UTKFace/NIQE_filter_64X64`. Run the filtering scripts `./run_test1.bat` and `./run_test2.bat` sequentially. A `.h5` file with Type II negative samples will be generated to `./fake_data`. Move this `.h5` file to `./UTKFace/UTKFace_64x64/CcGAN/baseline/output/CcGAN_SNGAN_soft_si0.041_ka3600.000_vanilla_nDs2_nDa1_nGa1_Dbs256_Gbs256/bad_fake_data/niters40000`. <br />
**Third**, go to the directory `./UTKFace/UTKFace_64x64/CcGAN/NDA`. Run the training script `./scripts/run_dual.bat` for Windows or `./scripts/run_dual.sh` for Linux. During this process, we train CcGAN with Dual-NDA for 60000 iterations with the Dual-NDA mechanism being activated starting after the first 40000 iterations.

* **Vanilla NDA** <br />
Go to the directory `./UTKFace/UTKFace_64x64/CcGAN/NDA`. Run the training script `./scripts/run_nda.bat` for Windows or `./scripts/run_nda.sh` for Linux.

* **ReACGAN and ADCGAN** <br />
Go to the directory `./UTKFace/UTKFace_64x64/class-conditional_GAN/StudioGAN`. Run the training script `./scripts/run_train1.bat` for ReACGAN or `./scripts/run_train2.bat` for ADCGAN. We conduct the evaluation process by running `./scripts/run_eval1.bat` for ReACGAN or `./scripts/run_eval2.bat` for ADCGAN.

* **ADM-G (Classifier Guidance)** <br />
Go to `./UTKFace/UTKFace_64x64/diffusion_models/ADM_G`. Then, run the script `./scripts/run_exp.bat`. 
    
* **CFG (Classifier-Free Guidance)** <br />
Go to `./UTKFace/UTKFace_64x64/diffusion_models/classifier-free_guidance`. Then, run the script `./scripts/run_exp.bat`. 


<!------------------------------------>
### (4) Steering Angle (128x128)










<!-- --------------------------------------------------------------- -->
## 4. Sampling and evaluation


### (1) SFID, Diversity, and Label Score


### (2) NIQE




<!-- -------------------------------
## References
[1] Ding, Xin, et al. "CcGAN: Continuous Conditional Generative Adversarial Networks for Image Generation." International Conference on Learning Representations. 2021.  <br />
[2] Lim, Jae Hyun, and Jong Chul Ye. "Geometric GAN." arXiv preprint arXiv:1705.02894 (2017).  <br />
[3] Zhang, Han, et al. "Self-attention generative adversarial networks." International conference on machine learning. PMLR, 2019.  <br />
[4] Zhao, Shengyu, et al. "Differentiable Augmentation for Data-Efficient GAN Training." Advances in Neural Information Processing Systems 33 (2020).  <br /> -->