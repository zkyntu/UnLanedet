# common setting
iou_loss_weight = 2.
cls_loss_weight = 2.
xyt_loss_weight = 0.2
seg_loss_weight = 1.0
num_points = 72
max_lanes = 4
sample_y = range(589, 230, -20)
test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)
ori_img_w = 1640
ori_img_h = 590
img_w = 800
img_h = 320
cut_height = 270
img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)
ignore_label = 255
bg_weight = 0.4
featuremap_out_channel = 192
num_classes = 4 + 1
data_root = "/root/autodl-tmp/culane"

SNR_SCALE = 2.0
SAMPLE_STEP = 1
HIDDEN_DIM = 256
DIM_DYNAMIC = 64
NUM_DYNAMIC = 2
POOLER_RESOLUTION = 7