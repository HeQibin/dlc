_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/uda_potirrg_to_vai_512x512.py',
    '../_base_/uda/dacs_a999_fdthings.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]
vol_weight=0.1
smooth_rate=0.02
batchsize=8
num_classes=6
scale=2
seed = 2  
model = dict(
    type='NTMCorrectEncoderDecoder',
    pretrained='/map-hl/qibin.he/pretrain_model/rs/mit_b5.pth',
    decode_head=dict(
        type='HRDAHeadWithNTMCorrect',
        single_scale_head='DAFormerHead',
        attention_classwise=True,
        hr_loss_weight=0.1,
        num_classes=num_classes,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0,
            smooth_rate=smooth_rate,
            ce_weight=0.8)
    ),
    scales=[1, 0.5],
    hr_crop_size=(256, 256),
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=True,
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[256, 256],
        crop_size=[512, 512]))
data = dict(
    train=dict(
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=2.0),
        target=dict(crop_pseudo_margins=[30, 240, 30, 30]),
    ),
    workers_per_gpu=1,
    samples_per_gpu=batchsize,
)
uda = dict(
    type='DACSWithNTMCorrect',
    ntm_loss_weight=0.2,
    ntm_all_point_cnt=15000,
    vol_weight=vol_weight,
    con_weight=0.1,
    ntm_lr=3e-4,
    ntm_lr_min=0,
    ntm_lr_pow=1.0,
    mask_mode='separatetrgaug',
    mask_alpha='same',
    mask_pseudo_threshold='same',
    mask_lambda=1,
    mask_generator=dict(
        type='block', mask_ratio=0.7, mask_block_size=64, _delete_=True))
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
gpu_model = 'NVIDIATESLAA100'
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=1)
evaluation = dict(interval=250, metric=['mIoU', 'mFscore'], save_best='mIoU')

name = f'potirrg2vai_dlc_smooth{smooth_rate}_vol{vol_weight}_bs8_10k'
exp = 'basic'
name_dataset = 'potirrg2vai_512x512'
name_architecture = 'hrda1-512-0.1_daformer_sepaspp_sl_mitb5'
name_encoder = 'mitb5'
name_decoder = 'hrda1-512-0.1_daformer_sepaspp_sl'
name_uda = f'dlc_smooth{smooth_rate}_vol{vol_weight}-ntmcorrect'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'

