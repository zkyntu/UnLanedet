# UnLanedet Model Zoo and Baselines

## Introduction

This file documents a collection of baselines trained with UnLanedet. All models are trained on the [AutoDL platform](https://www.autodl.com/) with a single 3090 GPU (or a 4090D GPU) with 24 GB memtory. We suggest that users train the model on this platform with the provided docker image.

### Tusimple baselines

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Venue</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Accuracy</th>
<th valign="bottom">Config</th>
<th valign="bottom">Weight</th>
<th valign="bottom">Log</th>
<tr><td align="center">SCNN</td>
<td align="center">AAAI</td>
<td align="center">ResNet18</td>
<td align="center">96.02</td>
<td align="center"><a href="../config/scnn/resnet18_tusimple.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/scnn_model_best_tusimple.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/scnn_log_tusimple.txt">train.log</a></td>
<tr><td align="center">RESA</td>
<td align="center">AAAI</td>
<td align="center">ResNet18</td>
<td align="center">96.27</td>
<td align="center"><a href="../config/resa/resnet18_tusimple.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/resa_model_best_tusimple.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/resa_log_tusimple.txt">train.log</a></td>
<tr><td align="center">UFLD</td>
<td align="center">ECCV</td>
<td align="center">ResNet18</td>
<td align="center">95.17</td>
<td align="center"><a href="../config/ufld/resnet18_tusimple.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/ufld_model_best_tusimple.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/ufld_log_tusimple.txt">train.log</a></td>
<tr><td align="center">CLRNet</td>
<td align="center">CVPR</td>
<td align="center">ResNet34</td>
<td align="center">96.64</td>
<td align="center"><a href="../config/clrnet/resnet34_tusimple.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_model_best_tusimple.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_log_tusimple.txt">train.log</a></td>
<tr><td align="center">LaneATT</td>
<td align="center">CVPR</td>
<td align="center">ResNet34</td>
<td align="center">94.65</td>
<td align="center"><a href="../config/laneatt/resnet18_tusimple.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/laneatt_model_best_tusimple.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/laneatt_log_tusimple.txt">train.log</a></td>
<tr><td align="center">ADNet</td>
<td align="center">ICCV</td>
<td align="center">ResNet34</td>
<td align="center">96.65</td>
<td align="center"><a href="../config/adnet/resnet34_tusimple.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/adnet_model_best_tusimple.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/adnet_log_tusimple.txt">train.log</a></td>
</tr>
<tr><td align="center">SRLane</td>
<td align="center">AAAI</td>
<td align="center">ResNet34</td>
<td align="center">96.21</td>
<td align="center"><a href="../config/srlane/resnet34_tusimple.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/srnet_r34_tusimple_model_best.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/srnet_r34_tusimple_log.txt">train.log</a></td>
</tr>
</tr>
<tr><td align="center">BezierNet</td>
<td align="center">CVPR</td>
<td align="center">ResNet18</td>
<td align="center">94.55</td>
<td align="center"><a href="../config/beziernet/resnet18_tusimple.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/beizernet_model_best.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/beziernet_tusimple_log.txt">train.log</a></td>
</tr>
</tbody></table>


### CULane baselines

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Venue</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">F1</th>
<th valign="bottom">Config</th>
<th valign="bottom">Weight</th>
<th valign="bottom">Log</th>
<tr><td align="center">UFLD</td>
<td align="center">ECCV</td>
<td align="center">ResNet18</td>
<td align="center">63.14</td>
<td align="center"><a href="../config/ufld/resnet18_culane.py">file</a></td>
<td align="center">-</td>
<td align="center">-</td>
<tr><td align="center">CLRNet</td>
<td align="center">CVPR</td>
<td align="center">ResNet34</td>
<td align="center">78.99</td>
<td align="center"><a href="../config/clrnet/resnet34_culane.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_r50_culane_model_best.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_r50_culane_log.txt">train.log</a></td>
<tr><td align="center">CLRNet</td>
<td align="center">CVPR</td>
<td align="center">ResNet50</td>
<td align="center">79.30</td>
<td align="center"><a href="../config/clrnet/resnet50_culane.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_model_best_culane.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrnet_log_culane.txt">train.log</a></td>
<tr><td align="center">CondLaneNet</td>
<td align="center">ICCV</td>
<td align="center">ResNet50</td>
<td align="center">79.69</td>
<td align="center"><a href="../config/condlane/resnet50_culane.py">file</a></td>
<td align="center">-</td>
<td align="center">-</td>
<tr><td align="center">CLRerNet</td>
<td align="center">WACV</td>
<td align="center">ResNet34</td>
<td align="center">79.20</td>
<td align="center"><a href="../config/clrernet/resnet34_culane.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrernet_model_best_culane.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/clrernet_log_culane.txt">train.log</a></td>
<tr><td align="center">ADNet</td>
<td align="center">ICCV</td>
<td align="center">ResNet34</td>
<td align="center">77.88</td>
<td align="center"><a href="../config/adnet/resnet34_culane.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/adnet_model_best_culane.pth">model.pth</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/adnet_log_culane.txt">train.log</a></td>
</tr>
</tbody></table>

### VIL100 baselines

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Venue</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">F1</th>
<th valign="bottom">Config</th>
<th valign="bottom">Weight</th>
<th valign="bottom">Log</th>
<tr><td align="center">ADNet</td>
<td align="center">ICCV</td>
<td align="center">ResNet34</td>
<td align="center">89.43</td>
<td align="center"><a href="../config/adnet/resnet34_vil.py">file</a></td>
<td align="center"><a href="https://github.com/zkyntu/UnLanedet/releases/download/Weights/adnet_model_final_vil100.pth">model.pth</a></td>
<td align="center">-</td>
</tr>
</tbody></table>

**Note**: 1) All models are trained from scratch. 2) Check the log file using the following codes: ```cat xxx.log``` (linux) or ```type xxx.log``` (windows). 3) The performance of the model is not fully aligned with the original paper due to the time limit.
