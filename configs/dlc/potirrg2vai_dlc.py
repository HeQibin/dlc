_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # Potsdam IRRG->Vaihingen Data Loading
    '../_base_/datasets/uda_potirrg_to_vai_512x512.py',
    # DAFormer Self-Training
    '../_base_/uda/dacs_a999_fdthings.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
vol_weight=0.1
smooth_rate=0.02
batchsize=8
num_classes=6
scale=2
# Random Seed
seed = 2  # seed with median performance
# HRDA Configuration
model = dict(
    type='NTMCorrectEncoderDecoder',
    pretrained='/map-hl/qibin.he/pretrain_model/rs/mit_b5.pth',
    decode_head=dict(
        type='HRDAHeadWithNTMCorrect',
        # Use the DAFormer decoder for each scale.
        single_scale_head='DAFormerHead',
        # Learn a scale attention for each class channel of the prediction.
        attention_classwise=True,
        # Set the detail loss weight $\lambda_d=0.1$.
        hr_loss_weight=0.1,
        num_classes=num_classes,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0,
            smooth_rate=smooth_rate,
            ce_weight=0.8)
    ),
    # Use the full resolution for the detail crop and half the resolution for
    # the context crop.
    scales=[1, 0.5],
    # Use a relative crop size of 0.5 (=512/1024) for the detail crop.
    hr_crop_size=(256, 256),
    # Use LR features for the Feature Distance as in the original DAFormer.
    feature_scale=0.5,
    # Make the crop coordinates divisible by 8 (output stride = 4,
    # downscale factor = 2) to ensure alignment during fusion.
    crop_coord_divisible=8,
    # Use overlapping slide inference for detail crops for pseudo-labels.
    hr_slide_inference=True,
    # Use overlapping slide inference for fused crops during test time.
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[256, 256],
        crop_size=[512, 512]))
data = dict(
    train=dict(
        # Rare Class Sampling
        # min_crop_ratio=2.0 for HRDA instead of min_crop_ratio=0.5 for
        # DAFormer as HRDA is trained with twice the input resolution, which
        # means that the inputs have 4 times more pixels.
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=2.0),
        # Pseudo-Label Cropping v2 (from HRDA):
        # Generate mask of the pseudo-label margins in the data loader before
        # the image itself is cropped to ensure that the pseudo-label margins
        # are only masked out if the training crop is at the periphery of the
        # image.
        target=dict(crop_pseudo_margins=[30, 240, 30, 30]),
    ),
    # Use one separate thread/worker for data loading.
    workers_per_gpu=1,
    # Batch size
    samples_per_gpu=batchsize,
)
# MIC Parameters
uda = dict(
    type='DACSWithNTMCorrect',
    ntm_loss_weight=0.2,
    ntm_all_point_cnt=15000,
    vol_weight=vol_weight,
    con_weight=0.1,
    ntm_lr=3e-4,
    ntm_lr_min=0,
    ntm_lr_pow=1.0,
    # imagenet feature distillation
    # imnet_feature_dist_lambda=0,  #0.005,
    # Apply masking to color-augmented target images
    # mask_mode=None,
    mask_mode='separatetrgaug',
    # Use the same teacher alpha for MIC as for DAFormer
    # self-training (0.999)
    mask_alpha='same',
    # Use the same pseudo label confidence threshold for
    # MIC as for DAFormer self-training (0.968)
    mask_pseudo_threshold='same',
    # Equal weighting of MIC loss
    mask_lambda=1,
    # Use random patch masking with a patch size of 64x64
    # and a mask ratio of 0.7
    mask_generator=dict(
        type='block', mask_ratio=0.7, mask_block_size=64, _delete_=True))
# Optimizer Hyperparameters
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
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=1)
evaluation = dict(interval=250, metric=['mIoU', 'mFscore'], save_best='mIoU')

# log_config = dict(interval=1)
# Meta Information for Result Analysis
name = f'potirrg2vai_dlc_smooth{smooth_rate}_vol{vol_weight}_bs8_10k'
exp = 'basic'
name_dataset = 'potirrg2vai_512x512'
name_architecture = 'hrda1-512-0.1_daformer_sepaspp_sl_mitb5'
name_encoder = 'mitb5'
name_decoder = 'hrda1-512-0.1_daformer_sepaspp_sl'
name_uda = f'dlc_smooth{smooth_rate}_vol{vol_weight}-ntmcorrect'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'

# For the other configurations used in the paper, please refer to experiment.py
