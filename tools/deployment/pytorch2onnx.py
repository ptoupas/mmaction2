# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint

from mmaction.models import build_model
from mmaction.models import recognizers
try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')

try:
    from mmcv.onnx.symbolic import register_extra_symbolics
except ModuleNotFoundError:
    raise NotImplementedError('please update mmcv to version>=1.0.4')


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def pytorch2onnx(model,
                 input_shape,
                 opset_version=14,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False):
    """Convert pytorch model to onnx model.

    Args:
        model (:obj:`nn.Module`): The pytorch model to be exported.
        input_shape (tuple[int]): The input tensor shape of the model.
        opset_version (int): Opset version of onnx used. Default: 14.
        show (bool): Determines whether to print the onnx model architecture.
            Default: False.
        output_file (str): Output onnx model name. Default: 'tmp.onnx'.
        verify (bool): Determines whether to verify the onnx model.
            Default: False.
    """
    model.cpu().eval()

    input_tensor = torch.randn(input_shape)
    if isinstance(model, recognizers.Recognizer2D):
        input_tensor = input_tensor.reshape((-1, ) + input_tensor.shape[2:])

    register_extra_symbolics(opset_version)
    torch.onnx.export(
        model,
        input_tensor,
        output_file,
        export_params=True,
        keep_initializers_as_inputs=False,
        verbose=show,
        opset_version=opset_version,
        do_constant_folding=False,
        training=torch.onnx.TrainingMode.PRESERVE)

    print(f'Successfully exported ONNX model: {output_file}')
    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_result = model(input_tensor)[0].detach().numpy()

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert len(net_feed_input) == 1
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(
            None, {net_feed_input[0]: input_tensor.detach().numpy()})[0]
        # only compare part of results
        random_class = np.random.randint(pytorch_result.shape[1])
        assert np.allclose(
            pytorch_result[:, random_class], onnx_result[:, random_class]
        ), 'The outputs are different between Pytorch and ONNX'
        print('The numerical values are same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMAction2 models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=14)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--is-localizer',
        action='store_true',
        help='whether it is a localizer')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 8, 224, 224],
        help='input video size')
    parser.add_argument(
        '--softmax',
        action='store_true',
        help='wheter to add softmax layer at the end of recognizers')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 14, 'MMAction2 only supports opset 14 now'

    cfg = mmcv.Config.fromfile(args.config)
    # import modules from string list.

    if not args.is_localizer:
        cfg.model.backbone.pretrained = None

    # build the model
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    model = _convert_batchnorm(model)

    # onnx.export does not support kwargs
    if hasattr(model, 'forward_dummy'):
        from functools import partial
        model.forward = partial(model.forward_dummy, softmax=args.softmax)
    elif hasattr(model, '_forward') and args.is_localizer:
        model.forward = model._forward
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # convert model to onnx file
    pytorch2onnx(
        model,
        args.shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify)

    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This tool will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)
