#!/bin/bash
#SBATCH --account=def-wjwelch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-05:00
#SBATCH --mail-user=dingx92@163.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=UK128_NIQE
#SBATCH --output=%x-%j.out


module load StdEnv/2020
module load cuda/11.4 python/3.9 opencv/4.5.5
module load matlab/2018a
virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements_narval.req


python imgs_to_groups_fake.py --imgs_dir ./fake_data/fake_images --out_dir_base ./fake_data

matlab -nodisplay -logfile output.txt -r "run niqe_test_steeringangle.m"