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
                ),
            anchor_base_size=300
        )
    )
)