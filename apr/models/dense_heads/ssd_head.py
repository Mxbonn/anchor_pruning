import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models import SSDHead as OGSSDHead
from mmdet.models.builder import HEADS


@HEADS.register_module(name=None, force=True, module=None)
class SSDHead(OGSSDHead):
    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        # TODO: Use registry to choose ConvModule type
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule

        for channel, num_base_priors in zip(self.in_channels,
                                            self.num_base_priors):
            cls_layers = []
            reg_layers = []
            in_channel = channel
            if num_base_priors > 0:
                # build stacked conv tower, not used in default ssd
                for i in range(self.stacked_convs):
                    cls_layers.append(
                        conv(
                            in_channel,
                            self.feat_channels,
                            3,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
                    reg_layers.append(
                        conv(
                            in_channel,
                            self.feat_channels,
                            3,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
                    in_channel = self.feat_channels
                # ssd-Lite head
                if self.use_depthwise:
                    cls_layers.append(
                        ConvModule(
                            in_channel,
                            in_channel,
                            3,
                            padding=1,
                            groups=in_channel,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
                    reg_layers.append(
                        ConvModule(
                            in_channel,
                            in_channel,
                            3,
                            padding=1,
                            groups=in_channel,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
                cls_layers.append(
                    nn.Conv2d(
                        in_channel,
                        num_base_priors * self.cls_out_channels,
                        kernel_size=1 if self.use_depthwise else 3,
                        padding=0 if self.use_depthwise else 1))
                reg_layers.append(
                    nn.Conv2d(
                        in_channel,
                        num_base_priors * 4,
                        kernel_size=1 if self.use_depthwise else 3,
                        padding=0 if self.use_depthwise else 1))
            self.cls_convs.append(nn.Sequential(*cls_layers))
            self.reg_convs.append(nn.Sequential(*reg_layers))

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
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs,
                                            self.cls_convs):
            if len(cls_conv) == 0:
                empty_size = list(feat.size())
                empty_size[1] = 0
                cls_scores.append(torch.empty(empty_size, device=feat.device))
                bbox_preds.append(torch.empty(empty_size, device=feat.device))
            else:
                cls_scores.append(cls_conv(feat))
                bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds
