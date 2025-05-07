import os.path as osp
import numpy as np
import math
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from scipy.interpolate import InterpolatedUnivariateSpline
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from omegaconf import DictConfig

def convert_dictconfig_to_dict(config):
    if isinstance(config, DictConfig):
        new_dict = {}
        for key, value in config.items():
            new_dict[key] = convert_dictconfig_to_dict(value)
        return new_dict
    else:
        return config

def CLRTransforms(img_h, img_w):
    return [
        dict(name='Resize',
             parameters=dict(size=dict(height=img_h, width=img_w)),
             p=1.0),
        dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
        dict(name='Affine',
             parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                    y=(-0.1, 0.1)),
                             rotate=(-10, 10),
                             scale=(0.8, 1.2)),
             p=0.7),
        dict(name='Resize',
             parameters=dict(size=dict(height=img_h, width=img_w)),
             p=1.0),
    ]

class GenerateLaneLine(object):
    def __init__(self, transforms=None, cfg=None, training=True):
        self.transforms = transforms
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.num_points = cfg.num_points
        self.n_offsets = cfg.num_points
        self.n_strips = cfg.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.max_lanes = cfg.max_lanes
        # import pdb;pdb.set_trace()
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        self.cfg = cfg
        self.training = training

        if transforms is None:
            transforms = CLRTransforms(self.img_h, self.img_w)

        if transforms is not None:
            transforms = [convert_dictconfig_to_dict(aug) for aug in transforms]
            img_transforms = []
            for aug in transforms:
                p = aug['p']
                if aug['name'] != 'OneOf':
                    img_transforms.append(
                        iaa.Sometimes(p=p,
                                      then_list=getattr(
                                          iaa,
                                          aug['name'])(**aug['parameters'])))
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf([
                                getattr(iaa,
                                        aug_['name'])(**aug_['parameters'])
                                for aug_ in aug['transforms']
                            ])))
        else:
            img_transforms = []
        self.transform = iaa.Sequential(img_transforms)

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def sample_lane(self, points, sample_ys):
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1],
                                              x[::-1],
                                              k=min(3,
                                                    len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y)
                                            & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain)

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1],
                            two_closest_points[:, 0],
                            deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))

        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane

    def transform_annotation(self, anno, img_wh=None):
        img_w, img_h = self.img_w, self.img_h

        old_lanes = anno['lanes']

        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates
        old_lanes = [[[
            x * self.img_w / float(img_w), y * self.img_h / float(img_h)
        ] for x, y in lane] for lane in old_lanes]
        # create tranformed annotations
        lanes = np.ones(
            (self.max_lanes, 2 + 1 + 1 + 2 + self.n_offsets), dtype=np.float32
        ) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, S+1 coordinates
        lanes_endpoints = np.ones((self.max_lanes, 2))
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.max_lanes:
                break

            try:
                xs_outside_image, xs_inside_image = self.sample_lane(
                    lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) <= 1:
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips
            lanes[lane_idx, 3] = xs_inside_image[0]

            thetas = []
            for i in range(1, len(xs_inside_image)):
                theta = math.atan(
                    i * self.strip_size /
                    (xs_inside_image[i] - xs_inside_image[0] + 1e-5)) / math.pi
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)

            theta_far = sum(thetas) / len(thetas)

            # lanes[lane_idx,
            #       4] = (theta_closest + theta_far) / 2  # averaged angle
            lanes[lane_idx, 4] = theta_far
            lanes[lane_idx, 5] = len(xs_inside_image)
            lanes[lane_idx, 6:6 + len(all_xs)] = all_xs
            lanes_endpoints[lane_idx, 0] = (len(all_xs) - 1) / self.n_strips
            lanes_endpoints[lane_idx, 1] = xs_inside_image[-1]

        new_anno = {
            'label': lanes,
            'old_anno': anno,
            'lane_endpoints': lanes_endpoints
        }
        return new_anno

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def __call__(self, sample):
        img_org = sample['img']
        # print(img_org is None)
        if 'cut_height' in sample.keys():
            self.cfg.cut_height = sample['cut_height']
        if self.cfg.cut_height != 0:
            new_lanes = []
            for i in sample['lanes']:
                lanes = []
                for p in i:
                    lanes.append((p[0], p[1] - self.cfg.cut_height))
                new_lanes.append(lanes)
            sample.update({'lanes': new_lanes})
        line_strings_org = self.lane_to_linestrings(sample['lanes'])
        line_strings_org = LineStringsOnImage(line_strings_org,
                                              shape=img_org.shape)

        # print(111111)
        for i in range(30):
            if self.training:
                mask_org = SegmentationMapsOnImage(sample['mask'],
                                                   shape=img_org.shape)
                # print("img:",(img_org is None))
                # print("line:",line_strings_org is None)
                # print("mask_org:",mask_org is None)
                img, line_strings, seg = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org,
                    segmentation_maps=mask_org)
                # print(44444)
                # print("img+++:",(img_org is None, img is None ))
                # print("line+++:",line_strings is None)
                # print("seg+++:",seg is None)
                # print("mask_org+++:",mask_org is None)
            else:
                img, line_strings = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org)
            line_strings.clip_out_of_image_()
            new_anno = {'lanes': self.linestrings_to_lanes(line_strings)}
            try:
                annos = self.transform_annotation(new_anno,
                                                  img_wh=(self.img_w,
                                                          self.img_h))
                label = annos['label']
                lane_endpoints = annos['lane_endpoints']
                break
            except:
                if (i + 1) == 30:
                    self.logger.critical(
                        'Transform annotation failed 30 times :(')
                    exit()

        # print(22222)
        sample['img'] = img.astype(np.float32) / 255.
        sample['lane_line'] = label
        sample['lanes_endpoints'] = lane_endpoints
        sample['gt_points'] = new_anno['lanes']
        sample['seg'] = seg.get_arr() if self.training else np.zeros(
            img_org.shape)

        return sample
    
class GenerateLaneLineATT(object):
    def __init__(self, transforms=None, wh=(640, 360), cfg=None):
        self.transforms = transforms
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.num_points = cfg.num_points
        self.n_offsets = cfg.num_points
        self.n_strips = cfg.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.max_lanes = cfg.max_lanes
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        transformations = iaa.Sequential([Resize({'height': self.img_h, 'width': self.img_w})])
        if transforms is not None:
            transforms = [getattr(iaa, aug['name'])(**convert_dictconfig_to_dict(aug)['parameters'])
                             for aug in transforms]  # add augmentation
        else:
            transforms = []
        self.transform = iaa.Sequential([iaa.Sometimes(then_list=transforms, p=1.0), transformations])

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def sample_lane(self, points, sample_ys):
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain)

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))

        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane

    def transform_annotation(self, anno, img_wh=None):
        img_w, img_h = self.img_w, self.img_h

        old_lanes = anno['lanes']

        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates
        old_lanes = [[[x * self.img_w / float(img_w), y * self.img_h / float(img_h)] for x, y in lane]
                     for lane in old_lanes]
        # create tranformed annotations
        lanes = np.ones((self.max_lanes, 2 + 1 + 1 + 1 + self.n_offsets),
                        dtype=np.float32) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 length, S+1 coordinates
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            try:
                xs_outside_image, xs_inside_image = self.sample_lane(lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) == 0:
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips
            lanes[lane_idx, 3] = xs_inside_image[0]
            lanes[lane_idx, 4] = len(xs_inside_image)
            lanes[lane_idx, 5:5 + len(all_xs)] = all_xs

        new_anno = {'label': lanes, 'old_anno': anno}
        return new_anno

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def __call__(self, sample):
        img_org = sample['img']
        line_strings_org = self.lane_to_linestrings(sample['lanes'])
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)

        for i in range(30):
            img, line_strings = self.transform(image=img_org.copy(), line_strings=line_strings_org)
            line_strings.clip_out_of_image_()
            new_anno = {'lanes': self.linestrings_to_lanes(line_strings)}
            try:
                label = self.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))['label']
                break
            except:
                if (i + 1) == 30:
                    self.logger.critical('Transform annotation failed 30 times :(')
                    exit()

        sample['img'] = (img / 255.).astype(np.float32)
        sample['lane_line'] = label

        return sample
    
    
class GenerateLanePts(object):
    def __init__(self, transforms=None, cfg=None, training=True):
        self.transforms = transforms
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.num_points = cfg.num_points
        self.n_offsets = cfg.num_points
        self.n_strips = cfg.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.max_lanes = cfg.max_lanes
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        self.cfg = cfg
        # TODO training should be in cfg!
        try:
            self.training = cfg.training
        except:
            self.training = training

        if transforms is None:
            transforms = CLRTransforms(self.img_h, self.img_w)

        if transforms is not None:
            transforms = [convert_dictconfig_to_dict(aug) for aug in transforms]
            img_transforms = []
            for aug in transforms:
                p = aug['p']
                if aug['name'] != 'OneOf':
                    img_transforms.append(
                        iaa.Sometimes(p=p,
                                      then_list=getattr(
                                          iaa,
                                          aug['name'])(**aug['parameters'])))
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf([
                                getattr(iaa,
                                        aug_['name'])(**aug_['parameters'])
                                for aug_ in aug['transforms']
                            ])))
        else:
            img_transforms = []
        self.transform = iaa.Sequential(img_transforms)

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def sample_lane(self, points, sample_ys):
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1],
                                              x[::-1],
                                              k=min(3,
                                                    len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y)
                                            & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain)

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1],
                            two_closest_points[:, 0],
                            deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))

        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane

    def transform_annotation(self, anno, img_wh=None):
        img_w, img_h = self.img_w, self.img_h

        old_lanes = anno['lanes']

        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates
        old_lanes = [[[
            x * self.img_w / float(img_w), y * self.img_h / float(img_h)
        ] for x, y in lane] for lane in old_lanes]
        # create tranformed annotations
        #===========================================================================
        # *                            plus theta
        lanes = np.ones(
            (self.max_lanes, 2 + 1 + 1 + 2 + self.n_offsets), dtype=np.float32
        ) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, S+1 coordinates
        #===========================================================================
        # lanes = np.ones((self.max_lanes, 2 + 1 + 1 + 1 + self.n_offsets),
        #                 dtype=np.float32) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 length, S+1 coordinates
        lanes_endpoints = np.ones((self.max_lanes, 2))
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.max_lanes:
                break

            try:
                xs_outside_image, xs_inside_image = self.sample_lane(
                    lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) <= 1:
                # print("debug")
                # _,_=self.sample_lane(lane, self.offsets_ys)
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = 1-(len(xs_outside_image) / self.n_strips)
            lanes[lane_idx, 3] = xs_inside_image[0]/self.img_w

            thetas = []
            for i in range(1, len(xs_inside_image)):
                theta = math.atan(
                    i * self.strip_size /
                    (xs_inside_image[i] - xs_inside_image[0] + 1e-5)) / math.pi
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)
            theta_far = sum(thetas) / len(thetas)

            # lanes[lane_idx,
            #       4] = (theta_closest + theta_far) / 2  # averaged angle

            #===========================================================================
            #  *                     plus theta,x,y,theta is 0~1!!!
            # TODO remove 180 for 0~1
            lanes[lane_idx, 4] = theta_far * 180
            lanes[lane_idx, 5] = len(xs_inside_image)
            lanes[lane_idx, 6:6 + len(all_xs)] = all_xs
            lanes_endpoints[lane_idx, 0] = (len(all_xs) - 1) / self.n_strips
            lanes_endpoints[lane_idx, 1] = xs_inside_image[-1] 
            #===========================================================================

            # lanes[lane_idx, 4] = len(xs_inside_image)
            # lanes[lane_idx, 5:5 + len(all_xs)] = all_xs
            # lanes_endpoints[lane_idx, 0] = (len(all_xs) - 1) / self.n_strips
            # lanes_endpoints[lane_idx, 1] = xs_inside_image[-1]

        new_anno = {
            'label': lanes,
            'old_anno': anno,
            'lane_endpoints': lanes_endpoints
        }
        return new_anno

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes
    def linestrings_to_gtpoints(self, lines):
        all_gt_points = []
        for line in lines:
            gt_points = []
            for points in line:
                gt_points.extend(list(points))
            all_gt_points.append(gt_points)
        return all_gt_points
    def __call__(self, sample):
        img_org = sample['img']
        if 'cut_height' in sample.keys():
            self.cfg.cut_height = sample['cut_height']
        if self.cfg.cut_height != 0:
            new_lanes = []
            for i in sample['lanes']:
                lanes = []
                for p in i:
                    lanes.append((p[0], p[1] - self.cfg.cut_height))
                new_lanes.append(lanes)
            sample.update({'lanes': new_lanes})
        line_strings_org = self.lane_to_linestrings(sample['lanes'])
        line_strings_org = LineStringsOnImage(line_strings_org,
                                              shape=img_org.shape)


        for i in range(30):
            if self.training:
                mask_org = SegmentationMapsOnImage(sample['mask'],
                                                   shape=img_org.shape)
                img, line_strings, seg = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org,
                    segmentation_maps=mask_org)
            else:
                img, line_strings = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org)
            line_strings.clip_out_of_image_()
            new_anno = {'lanes': self.linestrings_to_lanes(line_strings),
                        'gt_points': self.linestrings_to_gtpoints(line_strings)}
            try:
                annos = self.transform_annotation(new_anno,
                                                  img_wh=(self.img_w,
                                                          self.img_h))
                label = annos['label']
                lane_endpoints = annos['lane_endpoints']
                break
            except:
                if (i + 1) == 30:
                    self.logger.critical(
                        'Transform annotation failed 30 times :(')
                    exit()

        sample['img'] = img.astype(np.float32) / 255.
        sample['lane_line'] = label
        sample['lanes_endpoints'] = lane_endpoints
        sample['gt_points'] = new_anno['gt_points']
        sample['seg'] = seg.get_arr() if self.training else np.zeros(
            img_org.shape)

        return sample

class GenerateSRLaneLine(object):
    def __init__(self, transforms=None, cfg=None, training=True):
        self.transforms = transforms
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.num_points = cfg.num_points
        self.n_offsets = cfg.num_points
        self.n_strips = cfg.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.max_lanes = cfg.max_lanes
        self.feat_ds_strides = cfg.feat_ds_strides
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        self.training = training
        self.cfg = cfg

        if transforms is None:
            raise NotImplementedError("transforms is None")

        if transforms is not None:
            transforms = [convert_dictconfig_to_dict(aug) for aug in transforms]
            img_transforms = []
            for aug in transforms:
                p = aug["p"]
                if aug["name"] != "OneOf":
                    img_transforms.append(
                        iaa.Sometimes(p=p,
                                      then_list=getattr(
                                          iaa,
                                          aug["name"])(**aug["parameters"])))
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf([
                                getattr(iaa,
                                        aug_["name"])(**aug_["parameters"])
                                for aug_ in aug["transforms"]
                            ])))
        else:
            img_transforms = []
        self.transform = iaa.Sequential(img_transforms)

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    @staticmethod
    def sample_lane(points, sample_ys):
        """Interpolates the x-coordinates of a sorted set of points
        based on the given sample_ys.

        Args:
            points: Sorted points representing a lane.
            sample_ys:  Y-coordinates.

        Returns:
            ndarray: X-coordinates.
        """
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise ValueError("Annotaion points have to be sorted")
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1],
                                              x[::-1],
                                              k=min(3, len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y)
                                            & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(
            sample_ys_inside_domain)  # Since it is interpolation, the interp_xs are guaranteed to be within the range of the image. # noqa: E501

        # extrapolate lane to the bottom of the image with a straight line
        # using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1],
                            two_closest_points[:, 0],
                            deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)  # It is possible to exceed the range. # noqa: E501
        all_xs = np.hstack((extrap_xs, interp_xs))

        return all_xs

    @staticmethod
    def filter_duplicate_points(points):
        """Filters out duplicate points from a given list of points.

        Args:
            points: Sorted points representing a lane.

        Returns:
            List: Filtered points.
        """
        if points[-1][1] > points[0][1]:
            raise ValueError("Annotaion points have to be sorted")
        filtered_points = []
        used = set()
        for p in points:
            if p[1] not in used:
                filtered_points.append(p)
                used.add(p[1])

        return filtered_points

    @staticmethod
    def check_horizontal_lane(points, angle_threshold=5):
        """Check whether a lane is nearly horizontal.

        Args:
            points: Sorted points representing a lane.
            angle_threshold: angle threshold.

        Returns:
            bool: True if the lane angle is greater than the threshold,
             indicating a non-horizontal lane. False otherwise.
        """
        if len(points) < 2:
            return False
        rad = math.atan(
            math.fabs((points[-1][1] - points[0][1]) /
                      (points[0][0] - points[-1][0] + 1e-6)))
        angle = math.degrees(rad)

        return angle > angle_threshold

    def generate_angle_map(self, lanes):
        """Genrate ground-truth angle map for multi resolution features.

        Args:
            lanes: Annotatedd lanes.

        Returns:
            List: Angle maps.
        """
        gt_angle_list = []
        gt_seg_list = []
        for stride in self.feat_ds_strides:
            offsets_ys = np.arange(self.img_h, -1, -stride)
            gt_angle = np.zeros((self.img_h // stride, self.img_w // stride))
            gt_seg = np.zeros((self.img_h // stride, self.img_w // stride))
            for lane_idx, lane in enumerate(lanes, 1):
                try:
                    all_xs = self.sample_lane(
                        lane, offsets_ys)
                except AssertionError:
                    continue
                all_xs = all_xs / stride
                offsets_ys_down = offsets_ys / stride
                for i, (x, y) in enumerate(zip(all_xs[1:],
                                               offsets_ys_down[1:]), 1):
                    int_x, int_y = int(x), int(y)
                    if (int_x < 0 or int_x >= gt_angle.shape[1] or
                            int_y < 0 or int_y >= gt_angle.shape[0]):
                        continue
                    theta = math.atan(1 / (x - all_xs[i - 1] + 1e-6)) / math.pi
                    theta = theta if theta > 0 else 1 - abs(theta)
                    gt_angle[int_y][int_x] = theta
                    gt_seg[int_y][int_x] = 1  # lane_idx
            gt_angle_list.append(gt_angle)
            gt_seg_list.append(gt_seg)
        return gt_angle_list, gt_seg_list

    def transform_annotation(self, old_lanes):
        """Transforms the annotations.

        Args:
            old_lanes: Multi lanes represented by points.

        Returns:
            dict:
            - "gt_lane": Filtered, aligned, ground-truth lanes.
            - "gt_angle": Ground-truth lane angle map.
        """
        img_w, img_h = self.img_w, self.img_h
        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)  # The y-coordinate increases from the top of the image downwards. # noqa: E501
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_duplicate_points(lane) for lane in old_lanes]
        old_lanes = list(filter(self.check_horizontal_lane, old_lanes))
        # normalize the annotation coordinates
        old_lanes = [[[
            x * self.img_w / float(img_w), y * self.img_h / float(img_h)
        ] for x, y in lane] for lane in old_lanes]

        angle_map_list, seg_map_list = self.generate_angle_map(old_lanes)

        lanes = np.ones(
            (self.max_lanes, 2 + 2 + self.n_offsets), dtype=np.float32
        ) * -1e5  # 2 scores, 1 start_y, 1 length, n_offsets coordinates
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.max_lanes:
                break
            try:
                all_xs = self.sample_lane(
                    lane, self.offsets_ys)
            except AssertionError:
                continue
            # separate between inside and outside points
            inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
            xs_inside_image = all_xs[inside_mask]
            xs_outside_image = all_xs[~inside_mask]
            if len(xs_inside_image) <= 1:
                continue
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image)

            lanes[lane_idx, 3] = len(xs_inside_image)
            lanes[lane_idx, 4:4 + len(all_xs)] = all_xs

        new_anno = {
            "gt_lane": lanes,
            "gt_angle": angle_map_list,
            "gt_seg": seg_map_list,
        }
        return new_anno

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def __call__(self, sample):
        """Applies the lane transformation to a sample.

        Args:
            sample: The input sample containing "img" and "lanes" information.

        Returns:
            dict:
            - "img": Normalized image.
            - "lanes": Original lanes.
            - "gt_lane": Transformed, Filtered, aligned, ground-truth lanes.
            - "gt_angle": Ground-truth lane angle map.
        """
        if self.cfg.cut_height != 0:
            new_lanes = []
            for i in sample['lanes']:
                lanes = []
                for p in i:
                    lanes.append((p[0], p[1] - self.cfg.cut_height))
                new_lanes.append(lanes)
            sample.update({'lanes': new_lanes})
        img_org = sample["img"]
        line_strings_org = self.lane_to_linestrings(sample["lanes"])
        line_strings_org = LineStringsOnImage(line_strings_org,
                                              shape=img_org.shape)

        for i in range(10):
            if self.training:
                img, line_strings = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org)
            else:
                img, line_strings = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org)
            line_strings.clip_out_of_image_()
            try:
                annos = self.transform_annotation(
                    self.linestrings_to_lanes(line_strings))
                break
            except Exception as e:
                if (i + 1) == 10:
                    raise Exception(e)

        sample["img"] = img.astype(np.float32) / 255.
        sample.update(annos)

        return sample