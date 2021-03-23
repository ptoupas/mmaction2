# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(type='X3D', frozen_stages = 3, gamma_w=1, gamma_b=2.25, gamma_d=2.2),
    cls_head=dict(
        type='X3DHead',
        in_channels=432,
        num_classes=7,
        spatial_type='avg',
        dropout_ratio=0.85,
        fc1_bias=True),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
