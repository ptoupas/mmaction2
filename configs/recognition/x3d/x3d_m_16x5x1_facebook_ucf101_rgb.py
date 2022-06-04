_base_ = ["../../_base_/models/x3d.py"]
model = dict(
    type="Recognizer3D",
    backbone=dict(type="X3D", frozen_stages=-1, gamma_w=1, gamma_b=2.25, gamma_d=2.2),
    cls_head=dict(
        type="X3DHead",
        in_channels=432,
        num_classes=101,
        multi_class=False,
        spatial_type="avg",
        dropout_ratio=0.7,
        fc1_bias=False,
    ),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips="prob"),
)
# dataset settings
dataset_type = "VideoDataset"

data_root = "/data/datasets/ucf101/videos"
data_root_val = "/data/datasets/ucf101/videos"
split = 1
ann_file_train = f"/data/datasets/ucf101/ucf101_train_split_{split}_videos_mini.txt"
ann_file_val = f"/data/datasets/ucf101/ucf101_val_split_{split}_videos_mini.txt"
ann_file_test = f"/data/datasets/ucf101/ucf101_val_split_{split}_videos_mini.txt"

# dataset_type = 'RawframeDataset'
# data_root = 'data/hmdb51/rawframes'
# data_root_val = 'data/hmdb51/rawframes'
# ann_file_train = 'data/hmdb51/hmdb51_train_split_1_rawframes.txt'
# ann_file_val = 'data/hmdb51/hmdb51_val_split_1_rawframes.txt'

img_norm_cfg = dict(
    mean=[114.75, 114.75, 114.75], std=[57.38, 57.38, 57.38], to_bgr=False
)
train_pipeline = [
    dict(type="PyAVInit"),
    # dict(type='DecordInit'),
    dict(type="SampleFrames", clip_len=16, frame_interval=5, num_clips=1),
    # dict(type='DecordDecode'),
    dict(type="PyAVDecode"),
    # dict(type='RawFrameDecode'),
    dict(type="Resize", scale=(-1, 320)),
    dict(
        type="MultiScaleCrop",
        input_size=224,
        scales=(1, 0.875, 0.75),
        random_crop=False,
        max_wh_scale_gap=0,
    ),
    dict(type="ColorJitter"),
    dict(type="Resize", scale=(224, 224)),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs", "label"]),
]
val_pipeline = [
    dict(type="PyAVInit"),
    # dict(type='DecordInit'),
    dict(
        type="SampleFrames", clip_len=16, frame_interval=5, num_clips=1, test_mode=True
    ),
    # dict(type='RawFrameDecode'),
    # dict(type='DecordDecode'),
    dict(type="PyAVDecode"),
    dict(type="Resize", scale=(-1, 320)),
    dict(type="CenterCrop", crop_size=224),
    dict(type="Flip", flip_ratio=0),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs", "label"]),
]
test_pipeline = [
    dict(type="PyAVInit"),
    # dict(type='DecordInit'),
    dict(
        type="SampleFrames", clip_len=16, frame_interval=5, num_clips=1, test_mode=True
    ),
    # dict(type='RawFrameDecode'),
    # dict(type='DecordDecode'),
    dict(type="PyAVDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="CenterCrop", crop_size=256),
    # dict(type='ThreeCrop', crop_size=256),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs", "label"]),
]

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        multi_class=False,
        num_classes=101,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        multi_class=False,
        num_classes=101,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        multi_class=False,
        num_classes=101,
    ),
)
# optimizer
optimizer = dict(
    type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001
)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
# lr_config = dict(policy='fixed',
#                  warmup='linear',
#                  warmup_by_epoch=True,
#                  warmup_iters=5,
#                  warmup_ratio=0.05)
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_by_epoch=True,
    warmup_iters=3,
    warmup_ratio=0.05,
    step=15,
)
# lr_config = dict(policy='CosineAnnealing',
#                  min_lr=0,
#                  warmup='linear',
#                  warmup_by_epoch=True,
#                  warmup_iters=5,
#                  warmup_ratio=0.05)
# lr_config = dict(policy='CosineRestart', # https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9
#                  warmup='linear',
#                  warmup_by_epoch=True,
#                  warmup_iters=5,
#                  warmup_ratio=0.05,
#                  min_lr=0,
#                  periods=[10, 20],
#                  restart_weights=[1, 1])
total_epochs = 10
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=["top_k_accuracy", "mean_class_accuracy"])
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
    ],
)

# fp16 setting
# fp16 = dict()
# precise batchnormalization setting
# precise_bn = dict()
trt_model="checkpoints/trt_models/x3d_m.trt"

# runtime settings
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = "./work_dirs/tst/"
workflow = [("train", 1)]
# use the pre-trained model for the whole X3D-M network
load_from = (
    "checkpoints/x3d/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth"
)
resume_from = None
# set this True for multi-GPU training
find_unused_parameters = False
