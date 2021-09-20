# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(type='X3D', frozen_stages = -1, gamma_w=1, gamma_b=2.25, gamma_d=2.2),
    cls_head=dict(
        type='X3DHead',
        in_channels=432,
        num_classes=400,
        multi_class=False,
        spatial_type='avg',
        dropout_ratio=0.7,
        fc1_bias=False),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# model = dict(
#     type='Recognizer3D',
#     backbone=dict(type='X3D', frozen_stages = 0, gamma_w=2, gamma_b=2.25, gamma_d=5),
#     cls_head=dict(
#         type='X3DHead',
#         in_channels=864,
#         num_classes=7,
#         spatial_type='avg',
#         dropout_ratio=0.6,
#         fc1_bias=False),
#     # model training and testing settings
#     train_cfg=None,
#     test_cfg=dict(average_clips='prob'))
