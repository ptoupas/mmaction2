# _base_ = ['../../_base_/models/x3d.py']
model = dict(
    type='Recognizer3D',
    backbone=dict(type='X3D', frozen_stages = -1, gamma_w=1, gamma_b=2.25, gamma_d=2.2),
    cls_head=dict(
        type='X3DHead',
        in_channels=432,
        num_classes=16,
        multi_class=False,
        spatial_type='avg',
        dropout_ratio=0.8,
        fc1_bias=False),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
    # test_cfg=dict(average_clips='prob',num_clips=10,num_crops=3))

# dataset settings
dataset_type = 'VideoDataset'

bbox_folder_path = '/second_ext4/ptoupas/data/MSRDailyActivity3D/annotations'

data_root = f'/second_ext4/ptoupas/data/MSRDailyActivity3D/videos'
data_root_val = f'/second_ext4/ptoupas/data/MSRDailyActivity3D/videos'

ann_file_train = f'/second_ext4/ptoupas/data/MSRDailyActivity3D/msr_train_full.txt'
ann_file_val = f'/second_ext4/ptoupas/data/MSRDailyActivity3D/msr_test_full.txt'
ann_file_test = f'/second_ext4/ptoupas/data/MSRDailyActivity3D/msr_test_full.txt'

# ann_file_train = f'/second_ext4/ptoupas/data/MSRDailyActivity3D/msr_train_short.txt'
# ann_file_val = f'/second_ext4/ptoupas/data/MSRDailyActivity3D/msr_test_short.txt'
# ann_file_test = f'/second_ext4/ptoupas/data/MSRDailyActivity3D/msr_test_short.txt'

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
    dict(type='ApplyBbox'),
    dict(type='Resize', scale=(-1, 320)),
    dict(type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75),
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
    dict(type='ApplyBbox'),
    dict(type='Resize', scale=(-1, 320)),
    dict(type='CenterCrop', crop_size=224),
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
        clip_len=16,   
        frame_interval=10,
        num_clips=5,
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
    videos_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        multi_class=False,
        num_classes=16,
        bbox_ann_path=bbox_folder_path),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        multi_class=False,
        num_classes=16,
        bbox_ann_path=bbox_folder_path),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        multi_class=False,
        num_classes=16,
        bbox_ann_path=bbox_folder_path))
# optimizer
optimizer = dict(
    type='SGD', lr=0.013, momentum=0.9,
    weight_decay=0.0005)  # this lr is used for 8 gpus # try something like 0.0005 or 0.0001
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
# lr_config = dict(policy='step',
#                  warmup='linear',
#                  warmup_by_epoch=True,
#                  warmup_iters=5,
#                  warmup_ratio=0.1,
#                  step=[30]
#                  )
lr_config = dict(policy='CosineAnnealing',
                 min_lr=1e-5,
                #  by_epoch=True,
                #  warmup='linear',
                #  warmup_by_epoch=True,
                #  warmup_iters=5,
                #  warmup_ratio=0.1
                )
# lr_config = dict(policy='CosineRestart', # https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
#                  warmup='linear',
#                  warmup_by_epoch=True,
#                  warmup_iters=5,
#                  warmup_ratio=0.1,
#                  min_lr=0,
#                  periods=[10, 20],
#                  restart_weights=[1, 1])
total_epochs = 30
checkpoint_config = dict(interval=10)
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy',  'confusion_matrix'])
log_config = dict(
    interval=1,
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
work_dir = f'./work_dirs/x3d_m_msr/x3d_m_16x5x1_facebook_msr_bbox_rgb_v1'
workflow = [('train', 1)]
# use the pre-trained model for the whole X3D-M network
# load_from = 'checkpoints/x3d/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'
load_from = 'work_dirs/x3d_m_msr/x3d_m_16x5x1_facebook_msr_rgb_v6/epoch_60.pth'
resume_from = None

# set this True for multi-GPU training
find_unused_parameters=True

# find_unused_parameters = False
# gpu_ids = range(0, 1)
# omnisource = False
# module_hooks = []