# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead

import numpy as np
import torch
quant_fmaps = False
def quant_fmap(fmap, fractional_bits=10):
    # torch_to_numpy = fmap.cpu().numpy()
    fmap_shape = list(fmap.shape)

    shift_left = torch.ones(fmap_shape, dtype=torch.float32, device=torch.device('cuda:0'))*(2**fractional_bits)
    # shift_left = np.ones((fmap_shape))*(2**fractional_bits)
    # shift_left = shift_left.astype(np.float32)
    # shift_left = torch.from_numpy(shift_left).to(torch.device('cuda:0'))

    shift_right = torch.ones(fmap_shape, dtype=torch.float32, device=torch.device('cuda:0'))*(2**(-fractional_bits))
    # shift_right = np.ones((fmap_shape))*(2**(-fractional_bits))
    # shift_right = shift_right.astype(np.float32)
    # shift_right = torch.from_numpy(shift_right).to(torch.device('cuda:0'))

    fp_data = fmap * shift_left

    if fp_data.min() < -32768 or fp_data.max() > 32767:
        # print("Overflow on conversion to int16")
        # exit()
        # of_high = np.where(fp_data>32767)
        # fp_data[of_high] = 32767
        # of_low = np.where(fp_data<-32768)
        # fp_data[of_low] = -32767
        fp_data = fp_data.where(fp_data>=-32768.,torch.tensor(-32768.).to(torch.device('cuda:0')))
        fp_data = fp_data.where(fp_data<=32767.,torch.tensor(32767.).to(torch.device('cuda:0')))

    # fp_data = np.rint(fp_data).astype(np.short)
    fp_data = torch.round(fp_data).short()
    fp = fp_data * shift_right

    # res = torch.from_numpy(fq)
    # del torch_to_numpy, shift_left, shift_right, fp_data, fq
    del shift_left, shift_right, fp_data
    return fp

@HEADS.register_module()
class X3DHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        fc1_bias (bool): If the first fc layer has bias. Default: False.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 multi_class=False,
                 spatial_type='avg',
                 mid_channels = 2048,
                 dropout_ratio=0.5,
                 init_std=0.01,
                 fc1_bias=False):
        super().__init__(num_classes, in_channels, loss_cls, multi_class)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_classes = num_classes
        self.fc1_bias = fc1_bias

        self.fc1 = nn.Linear(
            self.in_channels, self.mid_channels, bias=self.fc1_bias)
        self.fc2 = nn.Linear(self.mid_channels, self.num_classes)

        self.relu = nn.ReLU()

        self.pool = None
        if self.spatial_type == 'avg':
            self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        elif self.spatial_type == 'max':
            self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        else:
            raise NotImplementedError

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc1, std=self.init_std)
        normal_init(self.fc2, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        if quant_fmaps:
            # [N, in_channels, T, H, W]
            assert self.pool is not None
            x = quant_fmap(x)
            x = self.pool(x)
            # [N, in_channels, 1, 1, 1]
            # [N, in_channels, 1, 1, 1]
            x = x.view(x.shape[0], -1)
            # [N, in_channels]
            x = quant_fmap(x)
            x = self.fc1(x)
            # [N, 2048]
            x = quant_fmap(x)
            x = self.relu(x)

            if self.dropout is not None:
                x = quant_fmap(x)
                x = self.dropout(x)

            x = quant_fmap(x)
            cls_score = self.fc2(x)
            cls_score = quant_fmap(cls_score)
            # [N, num_classes]
        else:
            # [N, in_channels, T, H, W]
            assert self.pool is not None
            x = self.pool(x)
            # [N, in_channels, 1, 1, 1]
            # [N, in_channels, 1, 1, 1]
            x = x.view(x.shape[0], -1)
            # [N, in_channels]
            x = self.fc1(x)
            # [N, 2048]
            x = self.relu(x)

            if self.dropout is not None:
                x = self.dropout(x)

            cls_score = self.fc2(x)
            # [N, num_classes]
        return cls_score
