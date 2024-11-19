# UnLanedet Model Zoo and Baselines

## Introduction

This file documents a collection of baselines trained with UnLanedet. All models are trained on the [AutoDL platform](https://www.autodl.com/) with a single 3090 GPU (or a 4090D GPU) with 24 GB memtory. We suggest that users train the model on this platform with the provided docker image.

We will upload the corresponding weights to facilitate the reproduction.

### Tusimple baselines

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Accuracy</th>
<th valign="bottom">Config</th>
<th valign="bottom">Weight</th>
<th valign="bottom">Log</th>
<tr><td align="center">SCNN</td>
<td align="center">ResNet18</td>
<td align="center">95.33</td>
<td align="center"><a href="../config/scnn/resnet18_tusimple.py">file</a></td>
<td align="center">-</td>
<td align="center">-</td>
<tr><td align="center">RESA</td>
<td align="center">ResNet18</td>
<td align="center">95.62</td>
<td align="center"><a href="../config/resa/resnet18_tusimple.py">file</a></td>
<td align="center">-</td>
<td align="center">-</td>
<tr><td align="center">UFLD</td>
<td align="center">ResNet18</td>
<td align="center">94.78</td>
<td align="center"><a href="../config/ufld/resnet18_tusimple.py">file</a></td>
<td align="center">-</td>
<td align="center">-</td>
<tr><td align="center">CLRNet</td>
<td align="center">ResNet34</td>
<td align="center">96.90</td>
<td align="center"><a href="../config/clrnet/resnet34_tusimple.py">file</a></td>
<td align="center">-</td>
<td align="center">-</td>
<tr><td align="center">LaneATT</td>
<td align="center">ResNet34</td>
<td align="center">94.65</td>
<td align="center"><a href="../config/laneatt/resnet18_tusimple.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/laneatt_model_best_tusimple.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/laneatt_log_tusimple.txt">train.log</a></td>
<tr><td align="center">ADNet</td>
<td align="center">ResNet34</td>
<td align="center">96.65</td>
<td align="center"><a href="../config/adnet/resnet34_tusimple.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/adnet_model_best_tusimple.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/adnet_log_tusimple.txt">train.log</a></td>
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
<th valign="bottom">Weight</th>
<th valign="bottom">Log</th>
<tr><td align="center">UFLD</td>
<td align="center">ResNet18</td>
<td align="center">63.14</td>
<td align="center"><a href="../config/ufld/resnet18_culane.py">file</a></td>
<td align="center">-</td>
<td align="center">-</td>
<tr><td align="center">CLRNet</td>
<td align="center">ResNet34</td>
<td align="center">78.99</td>
<td align="center"><a href="../config/clrnet/resnet34_culane.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_model_best_culane.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_log_culane.txt">train.log</a></td>
<tr><td align="center">CondLaneNet</td>
<td align="center">ResNet50</td>
<td align="center">79.69</td>
<td align="center"><a href="../config/condlane/resnet50_culane.py">file</a></td>
<td align="center">-</td>
<td align="center">-</td>
<tr><td align="center">CLRerNet</td>
<td align="center">ResNet34</td>
<td align="center">79.20</td>
<td align="center"><a href="../config/clrernet/resnet34_culane.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrernet_model_best_culane.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrernet_log_culane.txt">train.log</a></td>
<tr><td align="center">ADNet</td>
<td align="center">ResNet34</td>
<td align="center">77.88</td>
<td align="center"><a href="../config/adnet/resnet34_culane.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/adnet_model_best_culane.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/adnet_log_culane.txt">train.log</a></td>
</tr>
</tbody></table>

**Note**: All models are trained from scratch and reproduction results are different from the official repo. 
