# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import torch
from mmengine import Config
from mmengine.registry import init_default_scope

from mmaction.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 benchmark a recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--shape',
        required=True,
        type=int,
        nargs='+',
        help='input image size')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmaction'))

    model = MODELS.build(cfg.model)
    model = model.to("cuda")
    model.eval()

    batch_size, channels, frames, height, width = args.shape

    input_data = dict()
    input_tensor = torch.randn(1, batch_size, channels, frames, height, width).to("cuda")

    # the first several iterations may be very slow so skip them
    num_warmup = 10
    pure_inf_time = 0
    overall_fps = 0

    # benchmark with 2000 video and take the average
    for i in range(2100):

        input_data['inputs'] = input_tensor

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, **input_data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            overall_fps += fps
            if (i + 1) % args.log_interval == 0:
                print(
                    f'Done video [{i + 1:<3}/ 2000], fps: {fps:.2f} video / s, example inf time: {elapsed:.5f} ms, average inf time: {pure_inf_time / (i + 1 - num_warmup):.5f} ms')

        if (i + 1) == 2000:
            pure_inf_time += elapsed
            fps = overall_fps / (2000 - num_warmup)
            print(f'Overall fps: {fps:.5f} video / s, average inf time: {pure_inf_time / (i + 1 - num_warmup):.5f} ms')
            break


if __name__ == '__main__':
    main()
