import argparse
import math
from pprint import pprint

import apr
import mmcv
from mmdet.models import build_detector


def convert(args):
    print(f"Converting Anchor Generator of file {args.config}")
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    anchor_generator = cfg.model.bbox_head.anchor_generator
    anchor_generator_type = anchor_generator.type
    print(f"Converting generator of the type {anchor_generator_type} to PreciseAnchorGenerator.")

    pprint(anchor_generator, sort_dicts=False)

    print(f"Old config used:")
    scale_ratios = []
    anchor_base_size = args.s if args.s is not None else anchor_generator.strides[-1]
    new_config = {
        '_delete_': True,
        'type': 'PreciseAnchorGenerator',
        'strides': anchor_generator.strides,
        'scale_ratios': scale_ratios,
        'anchor_base_size': anchor_base_size
    }
    if anchor_generator_type == 'SSDAnchorGenerator':
        new_config['centers'] = [(stride/2, stride/2) for stride in anchor_generator.strides]

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    for base_anchors in model.bbox_head.prior_generator.base_anchors:
        scale_ratios_level = []
        for anchor_box in base_anchors:
            anchor_w = (anchor_box[2] - anchor_box[0])
            anchor_h = (anchor_box[3] - anchor_box[1])
            ratio = round((anchor_h / anchor_w).item(), 4)
            scale = round(((anchor_h / math.sqrt(ratio)) / anchor_base_size).item(), 4)
            scale_ratios_level.append((scale, ratio))
        scale_ratios.append(scale_ratios_level)

    print(f"New config to be used:")
    pprint(new_config, sort_dicts=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--s', type=int, help="Base size to which the size of anchors are expressed relatively. I.e. 300 for SSD300.")
    args = parser.parse_args()
    convert(args)
