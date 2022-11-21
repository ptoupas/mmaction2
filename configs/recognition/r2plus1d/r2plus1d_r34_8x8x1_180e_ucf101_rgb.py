_base_ = [
    '../../_base_/models/r2plus1d_r34.py', '../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/second_ext4/ptoupas/data/ucf101/videos'
data_root_val = '/second_ext4/ptoupas/data/ucf101/videos'
split = 3
ann_file_train = f'/second_ext4/ptoupas/data/ucf101/ucf101_train_split_{split}_videos.txt'
ann_file_val = f'/second_ext4/ptoupas/data/ucf101/ucf101_val_split_{split}_videos.txt'
ann_file_test = f'/second_ext4/ptoupas/data/ucf101/ucf101_val_split_{split}_videos.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='PyAVInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='PyAVDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='PyAVInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='PyAVDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='PyAVInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=10,
        test_mode=True),
    dict(type='PyAVDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=10,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        test_mode=True))
        
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[15, 30])
total_epochs = 40

# runtime settings
load_from = 'checkpoints/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb_20200618-3fce5629.pth'
checkpoint_config = dict(interval=5)
work_dir = f'./work_dirs/r2plus1d_r34_8x8x1_40e_ucf101_rgb_split_{split}/'
workflow = [('train', 1)]

dist_params = dict(backend='nccl')
log_level = 'INFO'

find_unused_parameters = True
gpu_ids = range(0, 1)
