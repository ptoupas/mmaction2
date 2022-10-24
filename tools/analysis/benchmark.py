# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
from matplotlib.pyplot import axis
import numpy as np

import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model

from tools.deployment.trt_engine import TrtModel, ONNXClassifierWrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 benchmark a recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--log-interval', default=10, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--trt-inference',
        action='store_true',
        help='Whether to use tensorrt engine for inference')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if not args.trt_inference:
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.backbone.pretrained = None
        cfg.data.test.test_mode = True

        # build the dataloader
        dataset = build_dataset(cfg.data.test, dict(test_mode=True))
        data_loader = build_dataloader(
            dataset,
            videos_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            persistent_workers=cfg.data.get('persistent_workers', False),
            dist=False,
            shuffle=False)

        # build the model and load checkpoint
        model = build_model(
            cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)

        model = MMDataParallel(model, device_ids=[0])

        model.eval()

        # the first several iterations may be very slow so skip them
        num_warmup = 5
        pure_inf_time = 0

        # benchmark with 2000 video and take the average
        for i, data in enumerate(data_loader):

            start_time = time.perf_counter()
            torch.cuda.synchronize()

            with torch.no_grad():
                model(return_loss=False, **data)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % args.log_interval == 0:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    print(
                        f'Done video [{i + 1:<3}/ 2000], fps: {fps:.1f} video / s'
                    )

            if (i + 1) == 200:
                pure_inf_time += elapsed
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Overall fps: {fps:.1f} video / s')
                break
    else:
        print('Using TensorRT engine for inference')
        num_classes = cfg.model['cls_head']['num_classes']
        # trt_model = TrtModel(
        #     cfg.trt_model, max_batch_size=1, dtype=np.float32)
        trt_model = ONNXClassifierWrapper(
            cfg.trt_model,
            num_classes=[1, num_classes],
            target_dtype=np.float32)

        # the first several iterations may be very slow so skip them
        num_warmup = 5
        pure_inf_time = 0

        # benchmark with 2000 video and take the average
        for i, data in enumerate(np.arange(2000)):

            start_time = time.perf_counter()

            # trt_data = data['imgs'].squeeze(
            #     axis=0).detach().numpy().astype(np.float32)
            # trt_model(trt_data, 1)
            trt_data = np.random.rand(1, 3, 16, 256, 256).astype(np.float32)
            trt_model.predict(trt_data)

            elapsed = time.perf_counter() - start_time

            if i >= num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % args.log_interval == 0:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    print(
                        f'Done video [{i + 1:<3}/ 2000], fps: {fps:.1f} video / s'
                    )

            if (i + 1) == 200:
                pure_inf_time += elapsed
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Overall fps: {fps:.1f} video / s')
                break


if __name__ == '__main__':
    main()
