mmdetection_configs_root = "/project_ghent/github/mmdetection/configs"

_base_ = [
    f"{mmdetection_configs_root}/ssd/ssd300_coco.py",
]
# model settings
model = dict(
    bbox_head=dict(
        anchor_generator=dict(
            _delete_=True,
            type="PreciseAnchorGenerator",
            strides=[8, 16, 32, 64, 100, 300],
            scale_ratios=(
                ((0.07, 1.0), (0.1024, 1.0)),
                ((0.15, 1.0), (0.15, 3.0), (0.15, 1 / 3), (0.224, 1.0)),
                ((0.33, 1.0), (0.33, 2.0), (0.33, 3.0), (0.33, 1 / 3), (0.4102, 1.0)),
                ((0.51, 3.0), (0.51, 1 / 3), (0.5932, 1.0)),
                ((0.69, 1.0), (0.69, 2.0)),
                ((0.9557, 1.0),),
            ),
            anchor_base_size=300,
            centers=((4, 4), (8, 8), (16, 16), (32, 32), (50, 50), (150, 150)),
        )
    )
)
