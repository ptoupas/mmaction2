_base_ = ['../../_base_/models/x3d.py']

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/home/active/ptoupas/external_hdd/Kinetics_700/kinetics700_2020/videos_train'
data_root_val = '/home/active/ptoupas/external_hdd/Kinetics_700/kinetics700_2020/videos_val' 
ann_file_train = '/home/active/ptoupas/external_hdd/Kinetics_700/kinetics700_2020/kinetics700_2020_train_list_videos.txt'
ann_file_val = '/home/active/ptoupas/external_hdd/Kinetics_700/kinetics700_2020/kinetics700_2020_val_list_videos.txt'

img_norm_cfg = dict(
    mean=[114.75, 114.75, 114.75], std=[57.38, 57.38, 57.38], to_bgr=False)
train_pipeline = [
    # dict(type='PyAVInit'),
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=5, num_clips=1),
    dict(type='DecordDecode'),
    # dict(type='PyAVDecode'),
    # dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 320)),
    dict(type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75),#, 0.625),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='ColorJitter'),
    dict(type='Resize', scale=(224, 224)),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    # dict(type='PyAVInit'),
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=1,
        test_mode=False),
    # dict(type='RawFrameDecode'),
    dict(type='DecordDecode'),
    # dict(type='PyAVDecode'),
    dict(type='Resize', scale=(-1, 320)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
# test_pipeline = [
#     dict(type='PyAVInit'),
#     # dict(type='DecordInit'),
#     dict(
#         type='SampleFrames',
#         clip_len=16,   
#         frame_interval=5,
#         num_clips=1,
#         test_mode=True),
#     # dict(type='RawFrameDecode'),
#     # dict(type='DecordDecode'),
#     dict(type='PyAVDecode'),
#     dict(type='Resize', scale=(-1, 320)),
#     dict(type='CenterCrop', crop_size=224),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['imgs'])
# ]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline))
    # test=dict(
    #     type=dataset_type,
    #     ann_file=ann_file_test,
    #     data_prefix=data_root_val,
    #     pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.05, momentum=0.9,
    weight_decay=0.0005)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
# lr_config = dict(policy='step',
#                  warmup='linear',
#                  warmup_by_epoch=True,
#                  warmup_iters=10,
#                  warmup_ratio=0.1,
#                  step=[35])
lr_config = dict(policy='CosineAnnealing',
                 min_lr=0,
                 warmup='linear',
                 warmup_by_epoch=True,
                 warmup_iters=13, # X3D original had 34 warmup epochs with 1024 batch size on a 256 total epochs run
                 warmup_ratio=0.1)
# lr_config = dict(policy='CosineRestart', # https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
#                  warmup='linear',
#                  warmup_by_epoch=True,
#                  warmup_iters=5,
#                  warmup_ratio=0.1,
#                  min_lr=0,
#                  periods=[10, 20],
#                  restart_weights=[1, 1])
total_epochs = 100 # X3D original was trained for 256 epochs with 1024 batch size
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=50,
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
work_dir = './work_dirs/x3d_m_16x5x1_facebook_kinetics700_2020_rgb_step_v1/'
workflow = [('train', 1)]
# use the pre-trained model for the whole X3D-M network
load_from = None #'checkpoints/x3d/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'
resume_from = None
# set this True for multi-GPU training
find_unused_parameters=True
