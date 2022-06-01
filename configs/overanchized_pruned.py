mmdetection_configs_root = '/home/mbonnaer/github.com/mmdetection/configs'

_base_ = [
    f'{mmdetection_configs_root}/ssd/ssd300_coco.py',
]
# model settings
model = dict(
    bbox_head=dict(
        anchor_generator=dict(
            _delete_=True,
            type='PreciseAnchorGenerator',
            strides=[8, 16, 32, 64, 100, 300],
            scale_ratios=(
                ((0.07, 1.0), (0.1025, 1.0), (0.07, 0.5), (0.07, 2.0)),
                ((0.15, 1.0), (0.2225, 1.0), (0.15, 0.5), (0.15, 2.0), (0.15, 1/3), (0.15, 3.0)),
                ((0.33, 1.0), (0.4102, 1.0), (0.33, 0.5), (0.33, 2.0), (0.33, 1/3), (0.33, 3.0)),
                ((0.51, 1.0), (0.5932, 1.0), (0.51, 0.5), (0.51, 2.0), (0.51, 1/3), (0.51, 3.0)),
                ((0.69, 1.0), (0.7748, 1.0), (0.69, 0.5), (0.69, 2.0)),
                ((0.87, 1.0), (0.9558, 1.0), (0.87, 0.5), (0.87, 2.0)),
                ),
            anchor_base_size=300
        )
    )
)