#!/bin/bash
CONFIG_FILE="/map-hl/qibin.he/results/rs/potirrg2vai/230915_1024_potirrg2vai_dacs_miou57.78/230915_1024_potirrg2vai_dacs_16e2c.py" 
CHECKPOINT_FILE="/map-hl/qibin.he/results/rs/potirrg2vai/230915_1024_potirrg2vai_dacs_miou57.78/best_mIoU_iter_3500.pth"
python3 -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU mFscore
