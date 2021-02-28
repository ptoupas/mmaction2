_base_ = [
    '../../_base_/models/tanet_r50.py', '../../_base_/default_runtime.py'
]

# dataset settings
# dataset_type = 'RawframeDataset'
# data_root = 'data/kinetics400/rawframes_train'
# data_root_val = 'data/kinetics400/rawframes_val'
# ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt'
# ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
# ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
dataset_type = 'RawframeDataset'
data_root = 'data/hmdb51/rawframes'
data_root_val = 'data/hmdb51/rawframes'
ann_file_train = 'data/hmdb51/hmdb51_train_split_1_rawframes.txt'
ann_file_val = 'data/hmdb51/hmdb51_val_split_1_rawframes.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='DenseSampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='DenseSampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=False),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
# test_pipeline = [
#     dict(
#         type='DenseSampleFrames',
#         clip_len=1,
#         frame_interval=1,
#         num_clips=8,
#         test_mode=True),
#     dict(type='RawFrameDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='ThreeCrop', crop_size=256),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCHW'),
#     dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['imgs'])
# ]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=6,
    # test_dataloader=dict(videos_per_gpu=1),
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
    #     pipeline=test_pipeline)

# optimizer
optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    lr=0.001,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_by_epoch=True,
                 warmup_iters=5,
                 warmup_ratio=0.1,
                 step=[20, 30])
total_epochs = 40

checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=3, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/tanet_r50_dense_1x1x8_100e_kinetics400_rgb/'
workflow = [('train', 1)]
# use the pre-trained model for the whole TANET network
load_from = 'checkpoints/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb_20210219-032c8e94.pth'
resume_from = None

