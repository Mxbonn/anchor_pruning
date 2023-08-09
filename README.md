# Anchor pruning for object detection [[`CVIU 2022`](https://doi.org/10.1016/j.cviu.2022.103445)]  [[`arXiv`](https://arxiv.org/abs/2104.00432)]

By Maxim Bonnaerens, Matthias Freiberger and Joni Dambre.

# Abstract

This paper proposes anchor pruning for object detection in one-stage anchor-based detectors.
While pruning techniques are widely used to reduce the computational cost of convolutional neural networks, they tend to
focus on optimizing the backbone networks where often most computations are. In this work we demonstrate an additional
pruning technique, specifically for object detection: anchor pruning.
With more efficient backbone networks and a growing trend of deploying object detectors on embedded systems where
post-processing steps such as non-maximum suppression can be a bottleneck, the impact of the anchors used in the
detection head is becoming increasingly more important.
In this work, we show that many anchors in the object detection head can be removed without any loss in accuracy. With
additional retraining, anchor pruning can even lead to improved accuracy. Extensive experiments on SSD and MS COCO show
that the detection head can be made up to 44% more efficient while simultaneously increasing accuracy. Further
experiments on RetinaNet and PASCAL VOC show the general effectiveness of our approach. We also introduce `
overanchorized' models that can be used together with anchor pruning to eliminate hyperparameters related to the initial
shape of anchors.

# Citation

```bibtex
@article{bonnaerens2022anchor,
  title={Anchor pruning for object detection},
  author={M. Bonnaerens, M. Freiberger and J. Dambre},
  journal={Computer Vision and Image Understanding},
  pages={103445},
  year={2022},
  publisher={Elsevier},
  doi = {https://doi.org/10.1016/j.cviu.2022.103445},
}
```

## Results and models of SSD

| Anchor Configuration              | AP .50:.95 | FLOPS head | BBoxes | Config                                                                                        | Download                                                                                             |
|-----------------------------------|------------|------------|--------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| SSD Baseline                      | 25.6       | 4231M      | 8732   | [config](https://github.com/Mxbonn/anchor_pruning/tree/master/configs/ssd/baseline.py)        | [model](https://cloud.ilabt.imec.be/index.php/s/ERxYsRMidPEgNwT)    |
| SSD Configuration-A retrained     | 25.4       | 3607M      | 7814   | [config](https://github.com/Mxbonn/anchor_pruning/tree/master/configs/ssd/configuration_A.py) | [model](https://cloud.ilabt.imec.be/index.php/s/ERxYsRMidPEgNwT) |
| **SSD Configuration-B retrained** | 25.6       | 2476M      | 4926   | [config](https://github.com/Mxbonn/anchor_pruning/tree/master/configs/ssd/configuration_B.py) | [model](https://cloud.ilabt.imec.be/index.php/s/ERxYsRMidPEgNwT) |
| SSD Configuration-C retrained     | 25.2       | 1628M      | 3121   | [config](https://github.com/Mxbonn/anchor_pruning/tree/master/configs/ssd/configuration_C.py) | [model](https://cloud.ilabt.imec.be/index.php/s/ERxYsRMidPEgNwT) |
| SSDConfiguration-D retrained      | 22.8       | 774M       | 1291   | [config](https://github.com/Mxbonn/anchor_pruning/tree/master/configs/ssd/configuration_D.py) | [model](https://cloud.ilabt.imec.be/index.php/s/ERxYsRMidPEgNwT) |
|    ||||||
| RetinaNet Baseline                | 36.5       | 129B       |    | [config](https://github.com/Mxbonn/anchor_pruning/tree/master/configs/retinanet/baseline.py)  | [model](https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth) |
| RetinaNet Pruned                  | 34.8       | **31B**    |    | [config](https://github.com/Mxbonn/anchor_pruning/tree/master/configs/retinanet/pruned.py)    | [model](https://cloud.ilabt.imec.be/index.php/s/ERxYsRMidPEgNwT) |



*Above results are on the COCO validation set while the results in the paper are on the COCO test set.*

![Results plot from paper](https://ars.els-cdn.com/content/image/1-s2.0-S1077314222000601-gr3.jpg)


## Installation

This repository builds upon [MMDetection](https://github.com/open-mmlab/mmdetection). 

See [The MMDetection documentation](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation instructions.
Last confirmed working version is mmdet v2.25.0 with mmcv-full v1.4.8
```bash
pip install openmim
mim install mmcv-full==1.4.8
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git switch --detach v2.25.0
pip install -v -e .
```

Next, clone our repository and install the anchor pruning package
```bash
git clone https://github.com/Mxbonn/anchor_pruning.git
cd anchor_pruning
pip install -e .
```


## Getting started.
Please see [`Tutorial.ipynb`](Tutorial.ipynb) for a general guide on how to do anchor pruning.

To run the given pretrained models above run
```bash
python tools/mmdet_test.py configs/ssd/configuration_B.py pretrained_models/configuration_B.pth --eval bbox
```
after modifying the paths to the mmdet base config in `configuration_X.py` and linking your dataset directory to `data/` similarly as required for mmdetection.
