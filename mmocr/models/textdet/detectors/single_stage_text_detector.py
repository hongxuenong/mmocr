# Copyright (c) OpenMMLab. All rights reserved.
from fileinput import filename
import torch

from mmocr.models.builder import DETECTORS
from mmocr.models.common.detectors import SingleStageDetector

import cv2


@DETECTORS.register_module()
class SingleStageTextDetector(SingleStageDetector):
    """The class for implementing single stage text detector."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        SingleStageDetector.__init__(self, backbone, neck, bbox_head,
                                     train_cfg, test_cfg, pretrained, init_cfg)

    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        preds = self.bbox_head(x)
        losses = self.bbox_head.loss(preds, **kwargs)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        # early return to avoid post processing
        if torch.onnx.is_in_onnx_export():
            return outs

        if len(img_metas) > 1:
            boundaries = [
                self.bbox_head.get_boundary(*(outs[i].unsqueeze(0)),
                                            [img_metas[i]], rescale)
                for i in range(len(img_metas))
            ]

        else:
            boundaries = [
                self.bbox_head.get_boundary(*outs, img_metas, rescale)
            ]

        ## debug and test
        # print(img_metas)
        # print(boundaries)
        # print(outs.shape)
        # filename = boundaries[0]['filename']
        # img_name = filename.split('/')[-1]
        # print(img.shape)
        # out_name = 'outputs/DBNet_CTW/test/' + img_name
        # cv2.imwrite(out_name, img[0].cpu().numpy().transpose(1, 2, 0) * 255)
        # out_name = 'outputs/viz/test/' + img_name.split(
        #     '.')[0] + '_prob' + '.png'
        # cv2.imwrite(out_name, outs[0, 0, :, :].cpu().numpy() * 255)
        # out_name = 'outputs/viz/test/' + img_name.split(
        #     '.')[0] + '_thrs' + '.png'
        # cv2.imwrite(out_name, outs[0, 1, :, :].cpu().numpy() * 255)
        # out_name = 'outputs/viz/test/' + img_name.split(
        #     '.')[0] + '_binary' + '.png'
        # cv2.imwrite(out_name, outs[0, 2, :, :].cpu().numpy() * 255)
        return boundaries
