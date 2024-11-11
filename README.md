# UnLanedet
<font size=4> An Unified 2D and 3D lane detection toolbox. UnLanedet contains many advanced lane detection methods to facilitate scientific research and lane detection applications. If you are in china, [gitee](https://gitee.com/zkyseured/UnLanedet) link may be helpful for you.

<div align="center">
  <img src="doc/Lane_Detection_Demo.jpg"/>
</div>
<br>

## What's New
* <font size=3> [2024-11-07] We release CondLaneNet and CLRerNet and fix bugs in UnLanedet. Try it!
* <font size=3> [2024-11-05] We release the v1 version, focusing on 2D lane detection methods.

## Installation
<font size=3> See [installation instructions](doc/install.md).

## Getting Started
<font size=3> See [Get Started documentation](scripts/TRAIN.md), including the data preparation, the traing code, the evaluation code, the resume code, and the inference code.

## Model Zoo and Baselines
We provide a set of lane detection methods, including segmentation-based and anchor-based. All models can be found in the [Model Zoo](doc/model_zpp.md).

## License
UnLanedet is released under the Apache 2.0 license.

## Acknowledgement
UnLanedet is built upon [detectron2](https://github.com/facebookresearch/detectron2), [lanedet](https://github.com/Turoad/lanedet) and [PPLanedet](https://github.com/zkyseu/PPlanedet). Many thanks to their great work!

## Citing UnLanedet
If you use UnLanedet in your research, please use the following BibTeX entry.

```BibTeX
@misc{zhouunlanedet,
  author =       {Kunyang Zhou},
  title =        {UnLanedet},
  howpublished = {\url{https://github.com/zkyntu/UnLanedet}},
  year =         {2024}
}
```
