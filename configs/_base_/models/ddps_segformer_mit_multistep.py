# -*- encoding: utf-8 -*-
'''
@File    :   ddps_segformer_mit_multistep.py
@Time    :   2023/09/19 15:25:33
@Author  :   Qibin He 
@Version :   1.0
@Contact :   qibin.he@outlook.com
'''

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DDPSEncoderDecoder',
    freeze_parameters=['backbone'],
    pretrained='/map-hl/qibin.he/pretrain_model/rs/mit_b5.pth',
    backbone=dict(type='mit_b5', style='pytorch'),
    decode_head=dict(
        type='SegformerHeadUnetFCHeadMultiStep',
        dim=256,
        out_dim=256,
        unet_channels=272,
        dim_mults=[1, 1, 1],
        cat_embedding_dim=16,
        diffusion_timesteps=100,
        collect_timesteps=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        ignore_index=0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
