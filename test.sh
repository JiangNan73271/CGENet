#!/bin/bash
source /home/ygq/anaconda3/etc/profile.d/conda.sh
conda activate zoomnext
#CUDA_VISIBLE_DEVICES=0 python main_for_image.py --config configs/icod_train.py --pretrained
CUDA_VISIBLE_DEVICES=1 python main_for_image.py --config configs/icod_train.py --model-name PvtV2B2_CGENet --evaluate --save-results --load-from ./model_pth/state_final.pth