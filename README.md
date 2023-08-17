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
Here, we provide a list of crucial software environments and python packages employed in the conducted experiments. Please note that we use different computational platforms for our experiments.

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

As illustrated in the aforementioned repository structure, distinct training codes have been provided for various datasets. For clarity, we will exemplify the training process using the **Steering Angle (64x64)** experiment. <br />
It's important to note that similar procedures can be followed for conducting experiments on other datasets.

### (0) Download checkpoints for evaluation models

Download the zip file of checkpoints of the evaluation models for Steering Angle (64x64) from [link](https://1drv.ms/u/s!Arj2pETbYnWQvMwHlZ362YyXnGuFXg?e=a0usQC). <br />
Then, unzip the file and copy the folder `eval_models` to `./Dual-NDA/SteeringAngle/SteeringAngle_64x64/evaluation`. <br />

We also provide the checkpoints of the evaluation models for other experiments as follows: <br />
[SteeringAngle_128x128_download_link](https://1drv.ms/u/s!Arj2pETbYnWQvMwIx6X1bMIVTKAj5Q?e=SD9P0S) <br />
[UTKFace_64x64_download_link](https://1drv.ms/u/s!Arj2pETbYnWQvMtFOZQSCMFuojGBmg?e=JsdVks) <br />
[UTKFace_128x128_download_link](https://1drv.ms/u/s!Arj2pETbYnWQvMwM7hKjcN0IVVRMPg?e=cbmCtP) <br />

### (1) Baseline CcGAN




### (2) Dual-NDA
#### (2.1) Generating Type II negative samples


#### (2.2) Train CcGANs with Dual-NDA



### (3) Vanilla NDA




### (4) ReACGAN and ADCGAN


ReACGAN.yaml is modified from https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/configs/Papa_ImageNet/ReACGAN.yaml

ADCGAN.yaml is modified from https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/master/src/configs/CIFAR100/ReACGAN-ADC-DiffAug.yaml


### (5) ADM-G (Classifier Guidance)




### (6) Classifier-Free Guidance




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