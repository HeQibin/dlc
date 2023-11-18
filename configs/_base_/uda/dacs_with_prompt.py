# -*- encoding: utf-8 -*-
'''
@File    :   dacs_with_prompt.py
@Time    :   2023/09/18 19:46:07
@Author  :   Qibin He 
@Version :   1.0
@Contact :   qibin.he@outlook.com
'''

# Baseline UDA
uda = dict(
    type='DACSWithPrompt',
    source_only=False,
    alpha=0.99,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    mask_mode=None,
    mask_alpha='same',
    mask_pseudo_threshold='same',
    mask_lambda=0,
    mask_generator=None,
    debug_img_interval=1000,
    print_grad_magnitude=False,
)
use_ddp_wrapper = True
