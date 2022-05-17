# Copyright (c) OpenMMLab. All rights reserved.
import torch
import cv2
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        x = self.extract_feat(imgs)
        if self.with_neck:
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if self.with_neck:
                    x, _ = self.neck(x)
                feats.append(x)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
        else:
            if 'num_clips' in self.test_cfg.keys() and 'num_crops' in self.test_cfg.keys():
                clips = self.test_cfg['num_clips']
                crops = self.test_cfg['num_crops']
                intermediate_feats = []
                for b in range(batches):
                    self.reset_gap_statistics = True
                    for ns in range(crops):
                        # Save images for validation of the input clips and frames flow
                        # imgs_new = imgs[ns*clips:ns*clips+clips,:].cpu().numpy()
                        # num_frames = imgs_new.shape[2]
                        # for cl in range(clips):
                        #     for fr in range(num_frames):
                        #         curr_image = imgs_new[cl,:,fr,:].transpose(1,2,0)
                        #         curr_image = cv2.cvtColor(curr_image, cv2.COLOR_RGB2BGR)
                        #         cv2.imwrite('/home/ptoupas/Development/mmaction2/img_{}_{}_{}.jpg'.format(ns, cl, fr), curr_image)
                        # print(f"Crop number {ns+1}")
                        # imgs_new = imgs[ns*clips:ns*clips+clips,:]
                        # self.reset_gap_statistics = True
                        for c in range(clips):
                            # print(f"Current clip: {b*crops*clips+ns*clips+c}")
                            imgs_new = imgs[b*crops*clips+ns*clips+c,:].unsqueeze(0)
                            feat_new = self.extract_feat(imgs_new)
                            self.reset_gap_statistics = False
                            intermediate_feats.append(feat_new)
                feat = torch.cat(intermediate_feats, 0)
                if self.with_neck:
                    feat, _ = self.neck(feat)
            else:
                feat = self.extract_feat(imgs)
                if self.with_neck:
                    feat, _ = self.neck(feat)                

        if self.feature_extraction:
            feat_dim = len(feat[0].size()) if isinstance(feat, tuple) else len(
                feat.size())
            assert feat_dim in [
                5, 2
            ], ('Got feature of unknown architecture, '
                'only 3D-CNN-like ([N, in_channels, T, H, W]), and '
                'transformer-like ([N, in_channels]) features are supported.')
            if feat_dim == 5:  # 3D-CNN architecture
                # perform spatio-temporal pooling
                avg_pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(feat, tuple):
                    feat = [avg_pool(x) for x in feat]
                    # concat them
                    feat = torch.cat(feat, axis=1)
                else:
                    feat = avg_pool(feat)
                # squeeze dimensions
                feat = feat.reshape((batches, num_segs, -1))
                # temporal average pooling
                feat = feat.mean(axis=1)
            return feat

        # should have cls_head if not extracting features
        assert self.with_cls_head
        cls_score = self.cls_head(feat)
        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        if len(imgs.shape) > 5:
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x, _ = self.neck(x)

        outs = self.cls_head(x)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)
