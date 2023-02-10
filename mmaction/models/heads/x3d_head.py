# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead

from fxpmath import Fxp
import numpy as np
import torch

quant_fmaps = True
if quant_fmaps:
    global_word_length = 11
    global_n_int = 5
    global_n_frac = 6

def quant_fmap(fmap, word_length=None, n_int=None, n_frac=None, signed=True):
    # Another convertion method    
    # fp_converter = Fxp(fmap.detach().cpu().numpy(), signed=True, n_word=global_word_length, n_frac=global_n_frac, n_int=global_n_int)
    # fp_tensor = fp_converter.get_val(dtype=fmap.detach().cpu().numpy().dtype)
    # fp_tensor = torch.from_numpy(fp_tensor).to(torch.device('cuda:0'))

    precision = 2**(-global_n_frac)
    rev_precision = 2**global_n_frac
    if signed:
        int_range = 2**(global_word_length -1)
        lower_val = - 2**(global_n_int -1)
    else:
        int_range = 2**(global_word_length)
        lower_val = 0
    upper_val = int_range - precision

    fmap_shape = list(fmap.shape)

    shift_left = torch.ones(fmap_shape, dtype=torch.float32, device=torch.device('cuda:0')) * rev_precision
    fp_data = fmap * shift_left

    fp_data = torch.trunc(fp_data)

    fp_data = torch.clip(fp_data, min=-int_range, max=int_range-1)

    shift_right = torch.ones(fmap_shape, dtype=torch.float32, device=torch.device('cuda:0')) * precision
    fp_data = fp_data * shift_right

    del shift_left, shift_right
    return fp_data

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
