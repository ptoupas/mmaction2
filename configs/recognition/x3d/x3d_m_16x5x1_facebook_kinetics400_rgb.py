_base_ = ['../../_base_/models/x3d.py']

# dataset settings
# dataset_type = 'RawframeDataset'
# data_root_val = 'data/kinetics400/rawframes_val'
# ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
dataset_type = 'RawframeDataset'
data_root = 'data/hmdb51/rawframes'
data_root_val = 'data/hmdb51/rawframes'
ann_file_train = 'data/hmdb51/hmdb51_train_split_1_rawframes.txt'
ann_file_val = 'data/hmdb51/hmdb51_val_split_1_rawframes.txt'
img_norm_cfg = dict(
    mean=[114.75, 114.75, 114.75], std=[57.38, 57.38, 57.38], to_bgr=False)
train_pipeline = [
    # dict(type='PyAVInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
        dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224)),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    # dict(type='PyAVInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
# test_pipeline = [
#     dict(type='DecordInit'),
#     dict(
#         type='SampleFrames',
#         clip_len=16,
#         frame_interval=5,
#         num_clips=10,
#         test_mode=True),
#     dict(type='RawFrameDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='CenterCrop', crop_size=256),
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
    type='SGD', lr=0.004, momentum=0.9,
    weight_decay=0.0005)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_by_epoch=True,
                 warmup_iters=5,
                 warmup_ratio=0.25,
                 step=[15, 25])
total_epochs = 30
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/x3d_m_16x5x1_facebook_kinetics400_rgb/'
workflow = [('train', 1)]
# use the pre-trained model for the whole X3D-M network
load_from = 'checkpoints/x3d/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'
resume_from = None
