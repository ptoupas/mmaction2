_base_ = ['../../_base_/models/x3d.py']

# dataset settings
dataset_type = 'VideoDataset'
# data_root = '/home/petros/Datasets/hmdb51/videos'
# data_root_val = '/home/petros/Datasets/hmdb51/videos'
# ann_file_train = '/home/petros/Datasets/hmdb51/hmdb51_train_split_1_videos.txt'
# ann_file_val = '/home/petros/Datasets/hmdb51/hmdb51_val_split_1_videos.txt'
# ann_file_test = '/home/petros/Datasets/hmdb51/hmdb51_val_split_1_videos.txt'
data_root = '/home/active/ptoupas/data/certhbot_har/videos'
data_root_val = '/home/active/ptoupas/data/certhbot_har/videos'
ann_file_train = '/home/active/ptoupas/data/certhbot_har/certhbot_har_train_split1.txt'
ann_file_val = '/home/active/ptoupas/data/certhbot_har/certhbot_har_val_split1.txt'
ann_file_test = '/home/active/ptoupas/data/certhbot_har/certhbot_har_val_split1.txt'

# dataset_type = 'RawframeDataset'
# data_root = 'data/hmdb51/rawframes'
# data_root_val = 'data/hmdb51/rawframes'
# ann_file_train = 'data/hmdb51/hmdb51_train_split_1_rawframes.txt'
# ann_file_val = 'data/hmdb51/hmdb51_val_split_1_rawframes.txt'
img_norm_cfg = dict(
    mean=[114.75, 114.75, 114.75], std=[57.38, 57.38, 57.38], to_bgr=False)
train_pipeline = [
    dict(type='PyAVInit'),
    # dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=13, frame_interval=6, num_clips=1),
    # dict(type='DecordDecode'),
    dict(type='PyAVDecode'),
    # dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 320)),
    dict(type='MultiScaleCrop',
        input_size=192,
        scales=(1, 0.875, 0.75),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='ColorJitter', color_space_aug=True),
    dict(type='Resize', scale=(192, 192)),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='PyAVInit'),
    # dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=13,
        frame_interval=6,
        num_clips=1,
        test_mode=False),
    # dict(type='RawFrameDecode'),
    # dict(type='DecordDecode'),
    dict(type='PyAVDecode'),
    dict(type='Resize', scale=(-1, 320)),
    dict(type='CenterCrop', crop_size=192),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(type='PyAVInit'),
    # dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=13,   
        frame_interval=6,
        num_clips=10,
        test_mode=True),
    # dict(type='RawFrameDecode'),
    # dict(type='DecordDecode'),
    dict(type='PyAVDecode'),
    dict(type='Resize', scale=(-1, 320)),
    dict(type='CenterCrop', crop_size=192),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9,
    weight_decay=0.0005)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
# lr_config = dict(policy='step',
#                  warmup='linear',
#                  warmup_by_epoch=True,
#                  warmup_iters=10,
#                  warmup_ratio=0.1,
#                  step=[30, 40])
lr_config = dict(policy='CosineAnnealing',
                 min_lr=0,
                 warmup='linear',
                 warmup_by_epoch=True,
                 warmup_iters=10,
                 warmup_ratio=0.1)
# lr_config = dict(policy='CosineRestart', # https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
#                  warmup='linear',
#                  warmup_by_epoch=True,
#                  warmup_iters=5,
#                  warmup_ratio=0.1,
#                  min_lr=0,
#                  periods=[10, 20],
#                  restart_weights=[1, 1])
total_epochs = 50
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

# fp16 setting
# fp16 = dict()
# precise batchnormalization setting
# precise_bn = dict()

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/x3d_s_16x5x1_facebook_hmdb7_rgb_step_v2/'
workflow = [('train', 1)]
# use the pre-trained model for the whole X3D-M network
load_from = 'checkpoints/x3d/x3d_s_facebook_13x6x1_kinetics400_rgb_20201027-623825a0.pth'
resume_from = None
find_unused_parameters=True