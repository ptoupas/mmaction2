_base_ = ['../../_base_/default_runtime.py']

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='TimeSformer',
        # pretrained=  # noqa: E251
        # 'https://download.openmmlab.com/mmaction/recognition/timesformer/vit_base_patch16_224.pth',  # noqa: E501
        num_frames=8,
        img_size=224,
        patch_size=16,
        embed_dims=768,
        in_channels=3,
        dropout_ratio=0.3,
        frozen_stages=5,
        transformer_layers=None,
        attention_type='divided_space_time',
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(type='TimeSformerHead', num_classes=7, in_channels=768),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# dataset settings
# dataset_type = 'RawframeDataset'
# data_root = 'data/kinetics400/rawframes_train'
# data_root_val = 'data/kinetics400/rawframes_val'
# ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt'
# ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
# ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
dataset_type = 'VideoDataset'
data_root = '/second_ext4/ptoupas/data/certhbot_har_fused_semi/videos'
data_root_val = '/second_ext4/ptoupas/data/certhbot_har_fused_semi/videos'
ann_file_train = '/second_ext4/ptoupas/data/certhbot_har_fused_semi/certhbot_har_train_split1_videos.txt'
ann_file_val = '/second_ext4/ptoupas/data/certhbot_har_fused_semi/certhbot_har_val_split1_videos.txt'
ann_file_test = '/second_ext4/ptoupas/data/certhbot_har_fused_semi/certhbot_har_val_split1_videos.txt'

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_bgr=False)

train_pipeline = [
    dict(type='PyAVInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
    # dict(type='RawFrameDecode'),
    dict(type='PyAVDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=224),
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
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    # dict(type='RawFrameDecode'),
    dict(type='PyAVDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(type='PyAVInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    # dict(type='RawFrameDecode'),
    dict(type='PyAVDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=8,
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
        pipeline=val_pipeline))#,
    # test=dict(
    #     type=dataset_type,
    #     ann_file=ann_file_test,
    #     data_prefix=data_root_val,
    #     pipeline=test_pipeline))

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    weight_decay=5e-4,
    nesterov=True)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[25, 40],
                 warmup='linear',
                 warmup_by_epoch=True,
                 warmup_iters=5,
                 warmup_ratio=0.1)
total_epochs = 50

# runtime settings
checkpoint_config = dict(interval=5)
load_from = 'checkpoints/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth'
work_dir = './work_dirs/timesformer_certhbot_har/timesformer_divST_8x32x1_15e_certhbot_rgb_v12'
