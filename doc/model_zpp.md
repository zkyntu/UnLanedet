# UnLanedet Model Zoo and Baselines

## Introduction

This file documents a large collection of baselines trained with UnLanedet. All models are trained on the [AutoDL platform](https://www.autodl.com/) with a single 3090 GPU (or a 4090D GPU) with 24 GB memtory. We suggest that users train the model on this platform with the provided docker image.

We will upload the corresponding weights to facilitate the reproduction.

### Tusimple baselines

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Accuracy</th>
<th valign="bottom">Config</th>
<tr><td align="center">SCNN</td>
<td align="center">ResNet50</td>
<td align="center">95.33</td>
<td align="center"><a href="../config/scnn/resnet50_tusimple.py">file</a></td>
<tr><td align="center">RESA</td>
<td align="center">ResNet18</td>
<td align="center">95.62</td>
<td align="center"><a href="../config/resa/resnet18_tusimple.py">file</a></td>
<tr><td align="center">UFLD</td>
<td align="center">ResNet18</td>
<td align="center">94.78</td>
<td align="center"><a href="../config/ufld/resnet18_tusimple.py">file</a></td>
<tr><td align="center">CLRNet</td>
<td align="center">ResNet34</td>
<td align="center">96.90</td>
<td align="center"><a href="../config/clrnet/resnet34_tusimple.py">file</a></td>
</tr>
</tbody></table>


### CULane baselines

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">F1</th>
<th valign="bottom">Config</th>
<tr><td align="center">UFLD</td>
<td align="center">ResNet18</td>
<td align="center">63.14</td>
<td align="center"><a href="../config/ufld/resnet18_culane.py">file</a></td>
<tr><td align="center">CLRNet</td>
<td align="center">ResNet34</td>
<td align="center">79.73</td>
<td align="center"><a href="../config/clrnet/resnet34_culane.py">file</a></td>
<tr><td align="center">CondLaneNet</td>
<td align="center">ResNet50</td>
<td align="center">79.69</td>
<td align="center"><a href="../config/condlane/resnet50_culane.py">file</a></td>


