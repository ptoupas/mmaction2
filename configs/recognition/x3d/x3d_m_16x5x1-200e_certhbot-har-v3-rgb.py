_base_ = ['../../_base_/models/x3d.py', '../../_base_/default_runtime.py']

model = dict(
    backbone=dict(pretrained=None),
    cls_head=dict(num_classes=7, dropout_ratio=0.5, average_clips='prob'))

version = 2
epochs = 300
warmup_epochs = 5
batch_size = 1

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/second_ext4/ptoupas/data/'
data_root_val = '/second_ext4/ptoupas/data/'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'/second_ext4/ptoupas/data/fused_har_train_split_{split}.txt'
ann_file_val = f'/second_ext4/ptoupas/data/fused_har_test_split_{split}.txt'
ann_file_test = f'/second_ext4/ptoupas/data/fused_har_test_split_{split}.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=16, frame_interval=5, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
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
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=5,
        num_clips=1,
        target_fps=30,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    # dict(type='ThreeCrop', crop_size=256),
    dict(type='CenterCrop', crop_size=256),
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
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric', metric_list=('top_k_accuracy', 'mean_class_accuracy'))
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=epochs, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# If you use the optimizer provided by Sophia, you need to upgrade mmengine to `0.7.4`.
# pip install Sophia-Optimizer
# optim_wrapper = dict(
#     optimizer=dict(type='SophiaG', lr=2e-4, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1),
#     clip_grad=dict(max_norm=40, norm_type=2))

# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.001, momentum=0.937, weight_decay=0.0005),
#     clip_grad=dict(max_norm=40, norm_type=2))
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=0.0003, betas=(0.9, 0.999), weight_decay=0.0005),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    # Linear learning rate warm-up scheduler
    dict(type='LinearLR',
         by_epoch=True,  # Updated by iterations
         begin=0,
         end=warmup_epochs),  # Warm up for the first 5 iterations
    # The main LRScheduler
    # dict(type='MultiStepLR',
    #      begin=warmup_epochs,
    #      end=epochs,
    #      by_epoch=True,  # Updated by epochs
    #      milestones=[50, 100],
    #      gamma=0.1)
    dict(type='CosineAnnealingLR',
         begin=warmup_epochs,
         end=epochs,
         by_epoch=True,
         T_max=epochs,
         eta_min=0,
         convert_to_iter_based=True)
]

default_hooks = dict(checkpoint=dict(interval=25),
                     logger=dict(type='LoggerHook', interval=5))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)
work_dir = f'/second_ext4/ptoupas/mmaction2/work_dirs/x3d_m_certhbot_har/x3d_m_16x5x1-{epochs}e_certhbot-har-v3-split_{split}_v{version}'

vis_backends = [dict(type='WandbVisBackend')]
                # dict(type='LocalVisBackend'),
                # dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=vis_backends,
)
load_from = 'checkpoints/x3d/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'
