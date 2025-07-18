# UnLanedet
<font size=4> An advanced lane detection toolbox. UnLanedet contains many advanced lane detection methods to facilitate scientific research and lane detection applications. If you are in China, [gitee](https://gitee.com/zkyseured/UnLanedet) link may be helpful for you.

<div align="center">
  <img src="doc/Lane_Detection_Demo.jpg"/>
</div>
<br>

## What's New 
* <font size=3> [2025-07-18] The latest distributed training code is provided. [PR link](https://github.com/zkyntu/UnLanedet/pull/53). Many thanks to the author.
* <font size=3> [2025-07-15] We release the paper of UnLanedet: [paper_link](doc/paper.pdf). Arxiv version is coming soon.
* <font size=3> [2025-05-30] We support DLA34 and ConvNexT backbone. CLRNet with ConvNext-Tiny gets 80.21 F1 score on CULane.
* <font size=3> [2025-05-27] We support GSENet and provide the [model analysis tools](./tools/analysis.py).
* <font size=3> [2025-05-23] We support GANet, a keypoint-based method, and modulated DCN in mmcv.
* <font size=3> [2025-05-14] We release v3 version. In this version, we support BezierNet, a parameter-based method.
* <font size=3> [2025-05-07] We support SRLane, a high-performance model with fast inference speed. Training on the custom dataset is provided in the advanced usage.
* <font size=3> [2025-04-24] We support distributed training (DDP) and provide the CLRNet-R50 model.
* <font size=3> [2025-03-12] We release the v2 version. In this version, we add the VIL100 dataset and the ADNet-VIL100 model and provide the [fps testing tool](./tools/test_speed.py). In the future, we will add O2SFormer, keypoint-based methods, and parameter-based methods. Stay tuned.
* <font size=3> [2025-03-04] We release the [Timm library wrapper](unlanedet/model/module/backbone/timm_wrapper.py)! Users can directly transfer the advanced backbone to UnLanedet. In the following weeks, we will release the v2 version.
* <font size=3> [2024-11-10] We release ADNet and LaneATT. Try it!
* <font size=3> [2024-11-07] We release CondLaneNet and CLRerNet and fix bugs in UnLanedet. Try it!
* <font size=3> [2024-11-05] We release the v1 version, focusing on 2D lane detection methods.

## Installation
<font size=3> See [installation instructions](doc/install.md).

## Getting Started
<font size=3> See [Get Started documentation](scripts/TRAIN.md), including the data preparation, the training code, the evaluation code, the resume code, the inference code, and the advanced usage.

## Model Zoo and Baselines
We provide a set of lane detection methods. All models and the corresponding weights and the training logs can be found in the [Model Zoo](doc/model_zpp.md).

## Advantages of UnLanedet
Compared with other lane detection libraries, e.g., lanedet and PPLanedet, UnLanedet has two obvious advantages: 1) Distributed training is supported. 2) More pretrained models and datasets are provided.

We do not depend on third-party library, such as mmcv series, and all modules and functions can be found in the repo.

## License
UnLanedet is released under the Apache 2.0 license.

## Contribution
We appreciate all contributions to UnLanedet and welcome pull requests to improve UnLanedet.

## Acknowledgement
UnLanedet is built upon [detectron2](https://github.com/facebookresearch/detectron2), [lanedet](https://github.com/Turoad/lanedet) and [PPLanedet](https://github.com/zkyseu/PPlanedet). Many thanks to their great work!

The code of BeizerNet is modified from [mmLaneDet](https://github.com/Yzichen/mmLaneDet). Many thanks to the authors.

Some modules are borrowed from [detrex](https://github.com/IDEA-Research/detrex). Many thanks to the authors.

## Citing UnLanedet
If you use UnLanedet in your research, please use the following BibTeX entry.

```BibTeX
@misc{zhouunlanedet,
  author =       {UnLanedet team},
  title =        {UnLanedet},
  howpublished = {\url{https://github.com/zkyntu/UnLanedet}},
  year =         {2024}
}
```
