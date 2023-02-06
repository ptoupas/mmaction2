# _base_ = ['../../_base_/models/x3d.py']
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='X3D', frozen_stages=-1, gamma_w=1, gamma_b=2.25, gamma_d=2.2),
    cls_head=dict(
        type='X3DHead',
        in_channels=432,
        num_classes=7,
        multi_class=False,
        spatial_type='avg',
        dropout_ratio=0.2,
        fc1_bias=False),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
# test_cfg=dict(average_clips='prob',num_clips=10,num_crops=3))

# dataset settings
dataset_type = 'VideoDataset'

split = 1

bbox_folder_path = '/second_ext4/ptoupas/data/certhbot_har_MSRDailyActivity3D_NTU_RGB/annotations'
data_root = f'/second_ext4/ptoupas/data/certhbot_har_MSRDailyActivity3D_NTU_RGB/videos'
data_root_val = f'/second_ext4/ptoupas/data/certhbot_har_MSRDailyActivity3D_NTU_RGB/videos'

ann_file_train = f'/second_ext4/ptoupas/data/certhbot_har_MSRDailyActivity3D_NTU_RGB/certhbot_har_msrdailyactivity3d_ntuRGB_train_split{split}_videos.txt'
ann_file_val = f'/second_ext4/ptoupas/data/certhbot_har_MSRDailyActivity3D_NTU_RGB/certhbot_har_msrdailyactivity3d_ntuRGB_val_split{split}_videos.txt'
ann_file_test = f'/second_ext4/ptoupas/data/certhbot_har_MSRDailyActivity3D_NTU_RGB/certhbot_har_msrdailyactivity3d_ntuRGB_val_split{split}_videos.txt'

# ann_file_train = f'/second_ext4/ptoupas/data/certhbot_har_MSRDailyActivity3D_NTU_RGB/certhbot_har_msrdailyactivity3d_train_split{split}_videos.txt'
# ann_file_val = f'/second_ext4/ptoupas/data/certhbot_har_MSRDailyActivity3D_NTU_RGB/certhbot_har_msrdailyactivity3d_val_split{split}_videos.txt'
# ann_file_test = f'/second_ext4/ptoupas/data/certhbot_har_MSRDailyActivity3D_NTU_RGB/certhbot_har_msrdailyactivity3d_val_split{split}_videos.txt'

# ann_file_train = f'/second_ext4/ptoupas/data/certhbot_har_MSRDailyActivity3D_NTU_RGB/certhbot_har_train_mini.txt'
# ann_file_val = f'/second_ext4/ptoupas/data/certhbot_har_MSRDailyActivity3D_NTU_RGB/certhbot_har_val_mini.txt'
# ann_file_test = f'/second_ext4/ptoupas/data/certhbot_har_MSRDailyActivity3D_NTU_RGB/certhbot_har_val_mini.txt'

img_norm_cfg = dict(
    mean=[114.75, 114.75, 114.75], std=[57.38, 57.38, 57.38], to_bgr=False)
train_pipeline = [
    dict(type='PyAVInit'),
    # dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=5, num_clips=1),
    # dict(type='DecordDecode'),
    dict(type='PyAVDecode'),
    # dict(type='RawFrameDecode'),
    dict(type='ColorJitter'),
    # dict(type='ApplyBbox'),
    dict(type='Resize', scale=(-1, 320)),
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
    dict(type='PyAVInit'),
    # dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=1,
        test_mode=True),
    # dict(type='RawFrameDecode'),
    # dict(type='DecordDecode'),
    dict(type='PyAVDecode'),
    # dict(type='ApplyBbox'),
    dict(type='Resize', scale=(-1, 256)),
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
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='CenterCrop', crop_size=256),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['imgs'])
# ]
test_pipeline = [
    dict(type='PyAVInit'),
    # dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=10,
        test_mode=True),
    # dict(type='RawFrameDecode'),
    # dict(type='DecordDecode'),
    dict(type='PyAVDecode'),
    # dict(type='ApplyBbox'),
    dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=256),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=10,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        multi_class=False,
        num_classes=7,
        bbox_ann_path=bbox_folder_path),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        multi_class=False,
        num_classes=7,
        bbox_ann_path=bbox_folder_path),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        multi_class=False,
        num_classes=7,
        bbox_ann_path=bbox_folder_path,
    ))

# optimizer SGD
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=0.00001
)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[30],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.1)

total_epochs = 65
checkpoint_config = dict(interval=10)
evaluation = dict(
    interval=5,
    metrics=['top_k_accuracy', 'mean_class_accuracy']) #, 'confusion_matrix'
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'/second_ext4/ptoupas/mmaction2/work_dirs/x3d_m_certhbot_har_MSRDailyActivity3D_NTU_RGB/x3d_m_16x5x1_facebook_certhbot_rgb_v5_split_{split}'
workflow = [('train', 1)]

# use the pre-trained model for the whole X3D-M network
load_from = 'checkpoints/x3d/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'
resume_from = None

# set this True for multi-GPU training
find_unused_parameters = False
gpu_ids = range(0, 1)

trt_model = "checkpoints/trt_models/x3d_m_7_fp32.engine"
