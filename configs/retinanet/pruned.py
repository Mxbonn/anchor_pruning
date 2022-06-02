mmdetection_configs_root = '/home/mbonnaer/github.com/mmdetection/configs'

_base_ = [
    f'{mmdetection_configs_root}/retinanet/retinanet_r50_fpn_1x_coco.py',
]
# model settings
model = dict(
    bbox_head=dict(
        anchor_generator=dict(
            _delete_=True,
            type='PreciseAnchorGenerator',
            strides=[8, 16, 32, 64, 128],
            scale_ratios=(
                (),
                ((0.125, 1.0), (0.125, 2.0), (0.125, 0.5),
                 (0.1575, 1.0), (0.1575, 2.0), (0.1575, 0.5),
                 (0.1984, 2.0), (0.1984, 0.5)),
                ((0.25, 1.0), (0.25, 2.0), (0.25, 0.5),
                 (0.315, 1.0), (0.315, 2.0), (0.315, 0.5),
                 (0.3969, 2.0), (0.3969, 0.5)),
                ((0.5, 1.0), (0.5, 2.0), (0.5, 0.5),
                 (0.63, 1.0), (0.63, 2.0), (0.63, 0.5),
                 (0.7937, 2.0), (0.7937, 0.5)),
                ((1.0, 1.0), (1.0, 2.0), (1.0, 0.5),
                 (1.26, 1.0), (1.26, 2.0), (1.26, 0.5),
                 (1.5874, 2.0), (1.5874, 0.5))

            ),
            anchor_base_size=512,
            centers=((4, 4), (8, 8), (16, 16), (32, 32), (64, 64))
            # For retraining, these centers do not need to be defined, but our checkpoint was trained with a slightly different code base.
        )
    )
)
