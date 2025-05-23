import cv2
import numpy as np
import scipy.interpolate as spi
from scipy.interpolate import InterpolatedUnivariateSpline
import random
import math

from .datacontainer import DataContainer as DC

from shapely.geometry import Polygon, Point, LineString, MultiLineString
import copy
import PIL


class GenerateGAInfo(object):
    def __init__(self,
                 radius=2,
                 fpn_cfg=dict(
                     hm_idx=0,
                     fpn_down_scale=[8, 16, 32],
                     sample_per_lane=[41, 21, 11],
                 ),
                 norm_shape=None,
                 cfg=None
                 ):
        self.radius = radius

        self.hm_idx = fpn_cfg.get('hm_idx')
        self.fpn_down_scale = fpn_cfg.get('fpn_down_scale')
        self.sample_per_lane = fpn_cfg.get('sample_per_lane')
        self.hm_down_scale = self.fpn_down_scale[self.hm_idx]
        self.fpn_layer_num = len(self.fpn_down_scale)
        self.cfg = cfg
        self.norm_shape = norm_shape

    def ploy_fitting_cube(self, line, h, w, sample_num=100):
        """
        Args:
            line: List[(x0, y0), (x1, y1), ...]    # y从大到小排列, 即lane由图像底部-->顶部
            h: f_H
            w: f_W
            sample_num: sample_per_lane
        Returns:
            key_points: (N_sample, 2)
        """
        line_coords = np.array(line).reshape((-1, 2))  # (N, 2)
        # y从小到大排列, 即lane由图像顶部-->底部
        line_coords = np.array(sorted(line_coords, key=lambda x: x[1]))

        lane_x = line_coords[:, 0]
        lane_y = line_coords[:, 1]

        if len(lane_y) < 2:
            return None
        new_y = np.linspace(max(lane_y[0], 0), min(lane_y[-1], h), sample_num)

        sety = set()
        nX, nY = [], []
        for i, (x, y) in enumerate(zip(lane_x, lane_y)):
            if y in sety:
                continue
            sety.add(x)
            nX.append(x)
            nY.append(y)
        if len(nY) < 2:
            return None

        if len(nY) > 3:
            ipo3 = spi.splrep(nY, nX, k=3)
            ix3 = spi.splev(new_y, ipo3)
        else:
            ipo3 = spi.splrep(nY, nX, k=1)
            ix3 = spi.splev(new_y, ipo3)
        return np.stack((ix3, new_y), axis=-1)

    def downscale_lane(self, lane, downscale):
        """
        :param lane: List[(x0, y0), (x1, y1), ...]
        :param downscale: int
        :return:
            downscale_lane: List[(x0/downscale, y0/downscale), (x1/downscale, y1/downscale), ...]
        """
        downscale_lane = []
        for point in lane:
            downscale_lane.append((point[0] / downscale, point[1] / downscale))
        return downscale_lane

    def clip_line(self, pts, h, w):
        pts_x = np.clip(pts[:, 0], 0, w - 1)[:, None]
        pts_y = np.clip(pts[:, 1], 0, h - 1)[:, None]
        return np.concatenate([pts_x, pts_y], axis=-1)

    def clamp_line(self, line, box, min_length=0):
        """
        Args:
            line: [(x0, y0), (x1,y1), ...]
            bbx: [0, 0, w-1, h-1]
            min_length: int
        """
        left, top, right, bottom = box
        loss_box = Polygon([[left, top], [right, top], [right, bottom],
                            [left, bottom]])
        line_coords = np.array(line).reshape((-1, 2))
        if line_coords.shape[0] < 2:
            return None
        try:
            line_string = LineString(line_coords)
            I = line_string.intersection(loss_box)
            if I.is_empty:
                return None
            if I.length < min_length:
                return None
            if isinstance(I, LineString):
                pts = list(I.coords)
                return pts
            elif isinstance(I, MultiLineString):
                pts = []
                Istrings = list(I)
                for Istring in Istrings:
                    pts += list(Istring.coords)
                return pts
        except:
            return None

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        """
        Args:
            heatmap: (hm_h, hm_w)   1/16
            center: (x0', y0'),  1/16
            radius: float
        Returns:
            heatmap: (hm_h, hm_w)
        """
        def gaussian2D(shape, sigma=1):
            """
            Args:
                shape: (diameter=2*r+1, diameter=2*r+1)
            Returns:
                h: (diameter, diameter)
            """
            m, n = [(ss - 1.) / 2. for ss in shape]
            # y: (1, diameter)    x: (diameter, 1)
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            # (diameter, diameter)
            h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            return h

        diameter = 2 * radius + 1
        # (diameter, diameter)
        gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def _transform_annotation(self, results):
        ori_img_h, ori_img_w = self.norm_shape
        img_h,img_w = self.cfg.img_h,self.cfg.img_w
        max_lanes = self.cfg.max_lanes

        gt_lanes = results['lanes']  # List[List[(x0, y0), (x1, y1), ...], List[(x0, y0), (x1, y1), ...], ...]
        
        scale_factor = ((img_w/ori_img_w),(img_h/ori_img_h))
        new_lanes = []
        for lane in results['lanes']:
            new_lane = []
            for p in lane:
                new_lane.append((p[0]*scale_factor[0], p[1]*scale_factor[1]))
            new_lanes.append(new_lane)
        gt_lanes = new_lanes

        # 遍历 fpn levels, 寻找每个车道线在该level特征图上对应采样点的位置.
        gt_hm_lanes = {}
        for l in range(self.fpn_layer_num):
            lane_points = []
            fpn_down_scale = self.fpn_down_scale[l]
            f_h = img_h // fpn_down_scale
            f_w = img_w // fpn_down_scale
            for i, lane in enumerate(gt_lanes):
                # downscaled lane: List[(x0, y0), (x1, y1), ...]
                lane = self.downscale_lane(lane, downscale=self.fpn_down_scale[l])
                # 将lane沿图像从下到上排列 （y由大到小）
                lane = sorted(lane, key=lambda x: x[1], reverse=True)
                pts = self.ploy_fitting_cube(lane, f_h, f_w, self.sample_per_lane[l])  # (N_sample, 2)
                if pts is not None:
                    pts_f = self.clip_line(pts, f_h, f_w)  # (N_sample, 2)
                    pts = np.int32(pts_f)
                    lane_points.append(pts[:, ::-1])  # (N_sample, 2)   2： (y, x)

            # (max_lane_num,  N_sample, 2)  2： (y, x)
            # 保存每个车道线在该level特征图上对应采样点的位置.
            lane_points_align = -1 * np.ones((max_lanes, self.sample_per_lane[l], 2))
            if len(lane_points) != 0:
                lane_points_align[:len(lane_points)] = np.stack(lane_points, axis=0)    # (num_lanes, N_sample, 2)
            gt_hm_lanes[l] = lane_points_align

        # 在最终所利用的level下，生成heatmap、offset等.
        # gt init
        hm_h = img_h // self.hm_down_scale
        hm_w = img_w // self.hm_down_scale
        kpts_hm = np.zeros((1, hm_h, hm_w), np.float32)     # (1, hm_H, hm_W)
        kp_offset = np.zeros((2, hm_h, hm_w), np.float32)   # (2, hm_H, hm_W)
        sp_offset = np.zeros((2, hm_h, hm_w), np.float32)   # (2, hm_H, hm_W)  key points -> start points
        kp_offset_mask = np.zeros((2, hm_h, hm_w), np.float32)  # (2, hm_H, hm_W)
        sp_offset_mask = np.zeros((2, hm_h, hm_w), np.float32)  # (2, hm_H, hm_W)

        start_points = []
        for i, lane in enumerate(gt_lanes):
            # downscaled lane: List[(x0, y0), (x1, y1), ...]
            lane = self.downscale_lane(lane, downscale=self.hm_down_scale)
            if len(lane) < 2:
                continue
            # (N_sample=int(360 / self.hm_down_scale), 2)
            lane = self.ploy_fitting_cube(lane, hm_h, hm_w, int(360 / self.hm_down_scale))
            if lane is None:
                continue
            # 将lane沿图像从下到上排列 （y由大到小）
            lane = sorted(lane, key=lambda x: x[1], reverse=True)
            lane = self.clamp_line(lane, box=[0, 0, hm_w - 1, hm_h - 1], min_length=1)
            if lane is None:
                continue

            start_point, end_point = lane[0], lane[-1]    # (2, ),  (2, )
            start_points.append(start_point)
            for pt in lane:
                pt_int = (int(pt[0]), int(pt[1]))   # (x, y)
                # 根据关键点坐标, 生成heatmap.
                self.draw_umich_gaussian(kpts_hm[0], pt_int, radius=self.radius)

                # 生成 compensation offsets, 对quantization error进行补偿.
                offset_x = pt[0] - pt_int[0]
                offset_y = pt[1] - pt_int[1]
                kp_offset[0, pt_int[1], pt_int[0]] = offset_x
                kp_offset[1, pt_int[1], pt_int[0]] = offset_y
                # 生成kp_offset_mask, 只有关键点位置处为1.
                kp_offset_mask[:, pt_int[1], pt_int[0]] = 1

                # 关键点到起始点之间的偏移
                offset_x = start_point[0] - pt_int[0]
                offset_y = start_point[1] - pt_int[1]
                sp_offset[0, pt_int[1], pt_int[0]] = offset_x
                sp_offset[1, pt_int[1], pt_int[0]] = offset_y
                # 生成kp_offset_mask, 只有关键点位置处为1.
                sp_offset_mask[:, pt_int[1], pt_int[0]] = 1

        targets = {}
        targets['gt_hm_lanes'] = gt_hm_lanes
        targets['gt_kpts_hm'] = kpts_hm
        targets['gt_kp_offset'] = kp_offset
        targets['gt_sp_offset'] = sp_offset
        targets['kp_offset_mask'] = kp_offset_mask
        targets['sp_offset_mask'] = sp_offset_mask

        return targets

    def __call__(self, results):
        targets = self._transform_annotation(results)
        results.update(targets)
#        print(results.keys())
        return results
