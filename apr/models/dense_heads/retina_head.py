import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models import RetinaHead as OGRetinaHead
from mmdet.models.builder import HEADS


@HEADS.register_module(name=None, force=True, module=None)
class RetinaHead(OGRetinaHead):
    def _init_layers(self):
        self.num_base_priors = max(self.prior_generator.num_base_priors)
        super(RetinaHead, self)._init_layers()
        self.num_base_priors = self.prior_generator.num_base_priors

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        cls_scores = []
        bbox_preds = []
        for feat, num_base_priors in zip(feats, self.num_base_priors):
            if num_base_priors == 0:
                empty_size = list(feat.size())
                empty_size[1] = 0
                empty_size[1] = 0
                cls_scores.append(torch.empty(empty_size, device=feat.device))
                bbox_preds.append(torch.empty(empty_size, device=feat.device))
            else:
                cls_score, bbox_pred = self.forward_single(feat)
                cls_scores.append(cls_score)
                bbox_preds.append(bbox_pred)
        return cls_scores, bbox_preds
