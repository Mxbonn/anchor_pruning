import warnings

import mmcv
import numpy as np
import torch
from torch.nn.modules.utils import _pair

from mmdet.core.anchor.builder import PRIOR_GENERATORS
from mmdet.core.anchor import AnchorGenerator


@PRIOR_GENERATORS.register_module()
class PreciseAnchorGenerator(AnchorGenerator):
    """Precise Anchor generator for 2D anchor-based detectors.
    This generator defines all individual anchors instead of combining ratio and scale lists,
    allowing more finetuned anchors.

    Args:
        strides (list[int] | list[tuple[int]]): Strides of anchors in multiple feature levels.
        scale_ratios (list[list[tuple(float]): list of list of (scale, aspect ratio) pairs with the
            outside list having the same number of entries as strides.
        anchor_base_size (int): Base size of the input to which the scales should be applied.
        centers (list[tuple[float, float]] | None): The centers of the anchor
            relative to the feature grid center in multiple feature levels.
            By default it is set to be None and not used. If a list of tuple of
            float is given, they will be used to shift the centers of anchors.
        center_offset (float): The offset of center in proportion to anchors'
            width and height. By default it is 0 in V2.0.
    """

    def __init__(self,
                 strides,
                 scale_ratios,
                 anchor_base_size,
                 centers=None,
                 center_offset=0.
                 ):
        # check center and center_offset
        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                                    f'!=0, {centers} is given.'
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'
        assert len(strides) == len(scale_ratios)

        self.strides = [_pair(stride) for stride in strides]
        self.anchor_base_size = anchor_base_size
        self.scale_ratios = scale_ratios

        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        assert len(scales) == len(ratios)
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        ws = (w * scales * w_ratios)
        hs = (h * scales * h_ratios)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    def gen_base_anchors(self):
        multi_level_base_anchors = []
        for i, level_scale_ratios in enumerate(self.scale_ratios):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            if not (level_scale_ratios and level_scale_ratios[0]):
                multi_level_base_anchors.append(torch.empty((0, 4)))
            else:
                scales, ratios = zip(*level_scale_ratios)
                scales = torch.tensor(scales)
                ratios = torch.tensor(ratios)
                base_anchors = self.gen_single_level_base_anchors(
                    self.anchor_base_size,
                    scales=scales,
                    ratios=ratios,
                    center=center
                )
                multi_level_base_anchors.append(base_anchors)
        return multi_level_base_anchors

    def __repr__(self):
        repr_str = self.__class__.__name__ + '()\n'
        return repr_str
