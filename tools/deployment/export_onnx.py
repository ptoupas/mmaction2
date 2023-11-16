# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import onnx
import onnxsim
import torch

from torch import nn
from onnxsim import simplify

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmaction.registry import RUNNERS


class CompleteModel(nn.Module):

    def __init__(self, base_model):
        super(CompleteModel, self).__init__()
        self.backbone = base_model.backbone
        self.head = base_model.cls_head

    def forward(self, input_tensor):
        feat = self.backbone(input_tensor)
        cls_score = self.head(feat)
        return cls_score


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 convert a model to onnx')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'output_file', help='file name of the output onnx file')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch size of inference')
    parser.add_argument(
        '--num_channels', type=int, default=3, help='num of input channels')
    parser.add_argument(
        '--num_frames', type=int, default=16, help='number of input frames.')
    parser.add_argument(
        '--img_shape', type=int, nargs='+', default=[224, 224], help='input image size')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--device', type=str, default='cpu', help='CPU/CUDA device option')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    base_model = runner.model
    model = CompleteModel(base_model).to(args.device)
    model.eval()

    model_name = os.path.basename(args.output_file).split('.')[0]

    input_shape = (args.batch_size, args.num_channels, args.num_frames, args.img_shape[0], args.img_shape[1])
    x = torch.randn(input_shape).to(args.device)
    
    torch.onnx.export(model,                # model being run
                      # model input (or a tuple for multiple inputs)
                      x,
                      # where to save the model (can be a file or file-like object)
                      f"{model_name}.onnx",
                      export_params=True,   # store the trained parameter weights inside the model file
                      opset_version=14,     # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['output'],  # the model's output names
                      keep_initializers_as_inputs=False,
                      )

    onnx_model = onnx.load(f"{model_name}.onnx")
    model_simp, check = onnxsim.simplify(onnx_model)
    onnx.helper.strip_doc_string(model_simp)
    model_simp = onnx.shape_inference.infer_shapes(model_simp)
    onnx.checker.check_model(model_simp)
    onnx.save(model_simp, f"{model_name}_simplified.onnx")


if __name__ == '__main__':
    main()
