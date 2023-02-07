_base_ = ['../../_base_/models/x3d.py', '../../_base_/default_runtime.py']

model = dict(
    backbone=dict(type='X3D', frozen_stages=-1, gamma_w=1, gamma_b=2.25, gamma_d=2.2),
    cls_head=dict(num_classes=7, average_clips='prob'))

# dataset settings
split = 1
dataset_type = 'VideoDataset'
data_root = '' #'data/kinetics400/videos_train'
data_root_val = '' #'data/kinetics400/videos_val'
ann_file_train = f'/second_ext4/ptoupas/data/fused_har_train_split_{split}.txt'
ann_file_val = f'/second_ext4/ptoupas/data/fused_har_test_split_{split}.txt'

epochs = 50
batch_size = 16

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=5, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))

test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=10,
        target_fps=30,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=256),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))


val_evaluator = dict(type='AccMetric', metric_list=('top_k_accuracy', 'mean_class_accuracy'))
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=epochs, val_begin=1, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning policy
param_scheduler = [
    # warmup
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    # main learning rate scheduler
    # dict(
    #     type='CosineAnnealingLR',
    #     eta_min=0,
    #     T_max=45,
    #     by_epoch=True,
    #     begin=5,
    #     end=epochs,
    #     convert_to_iter_based=True),
    dict(
        type='MultiStepLR',
        begin=5,
        end=epochs,
        by_epoch=True,
        milestones=[35],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)

# runtime settings
default_hooks = dict(
                     checkpoint=dict(interval=5, max_keep_ckpts=2),
                     logger=dict(type='LoggerHook', interval=10),)

work_dir = f'/second_ext4/ptoupas/mmaction2/work_dirs/x3d_m_16x5x1_facebook-kinetics400-pre-fused_har-rgb-split_{split}'

log_processor = dict(
      type='LogProcessor',
      window_size=20,
      by_epoch=True)

vis_backends = [
      dict(type='TensorboardVisBackend'),]
    #   dict(type='LocalVisBackend'),
    #   dict(type='WandbVisBackend')]
visualizer = dict(
    type='ActionVisualizer',
    save_dir=f"{work_dir}",
    vis_backends=vis_backends,
)

load_from = 'checkpoints/x3d/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'
