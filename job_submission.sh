#!/bin/bash
#SBATCH --job-name=PhIRE20
#SBATCH --partition=matador
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=40
#%module load matador/0.15.4
#%module load gcc/8.4.0
#%module load cudnn/7.6.5.32-10.2-linux-x64-cuda
#%module load cuda/10.2.89

. $HOME/conda/etc/profile.d/conda.sh
conda activate TF1.15
cd /home/guiyli/MyTmp/PhIRE

# python generate_TFRecord.py
# python main_new.py --data_type solar --dataset_name solar_2009 --batch_size 16 --epoch 10
# python main_new.py --data_type solar --dataset_name solar_2007,2008,2009,2010,2011,2012 --batch_size 16 --epoch 10
python main_new.py --data_type solar --dataset_name solar_2007,2008,2009,2010,2011,2012 --batch_size 16 --epoch 20

