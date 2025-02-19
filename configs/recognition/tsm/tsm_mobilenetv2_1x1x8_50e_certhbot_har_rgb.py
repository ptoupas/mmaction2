_base_ = [
    '../../_base_/models/tsm_mobilenet_v2.py',
    '../../_base_/schedules/sgd_tsm_mobilenet_v2_50e.py',
    '../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/data/Data/CERTHBOT_dataset/certhbot_har/rawframes'
data_root_val = '/data/Data/CERTHBOT_dataset/certhbot_har/rawframes'
ann_file_train = '/data/Data/CERTHBOT_dataset/certhbot_har/certhbot_har_train_split1_rawframes.txt'
ann_file_val = '/data/Data/CERTHBOT_dataset/certhbot_har/certhbot_har_val_split1_rawframes.txt'
# ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DenseSampleFrames', clip_len=1, frame_interval=2, num_clips=8),
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
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
# test_pipeline = [
#     dict(
#         type='SampleFrames',
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
    videos_per_gpu=1,
    workers_per_gpu=1,
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
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    lr=0.001,  # this lr is used for 8 gpus
)

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = './work_dirs/tsm_mobilenetv2_dense_1x1x8_100e_certhbot_har_rgb/'
