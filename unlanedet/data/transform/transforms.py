import math
import random
import cv2
import numpy as np
import torch
import numbers
import collections
from PIL import Image
from collections.abc import Sequence
from .datacontainer import DataContainer as DC 

try:
    collections = collections.abc
except:
    pass

def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

class DCToTensor(object):
    def __init__(self, keys=['img', 'mask'], collect_keys=[], cfg=None):
        self.keys = keys
        self.collect_keys = collect_keys

    def __call__(self, sample):
        data = {}
        if len(sample['img'].shape) < 3:
            sample['img'] = np.expand_dims(sample['img'], -1)
        for key in sample.keys():
            if isinstance(sample[key], list) or isinstance(sample[key], dict):
                data[key] = sample[key]
                continue
            if key in self.keys:
                data[key] = DC(to_tensor(sample[key]),stack=False)
            if key in self.collect_keys:
                data[key] = sample[key]
        data['img'] = data['img'].permute(2, 0, 1)
        return data
    
class ListToTensor(object):
    def __init__(self, keys=['img', 'mask'], collect_keys=[], cfg=None):
        self.keys = keys
        self.collect_keys = collect_keys

    def __call__(self, sample):
        data = {}
        if len(sample['img'].shape) < 3:
            sample['img'] = np.expand_dims(sample['img'], -1)
        for key in sample.keys():
            if isinstance(sample[key], list) or isinstance(sample[key], dict):
                data[key] = sample[key]
                continue
            if key in self.keys:
                if isinstance(sample[key],DC):
                    data[key] = sample[key]
                else:
                    data[key] = to_tensor(sample[key])
            if key in self.collect_keys:
                data[key] = sample[key]
        data['img'] = data['img'].permute(2, 0, 1)
        return data

    
class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
        collect_keys (Sequence[str]): Keys that need to keep, but not to Tensor.
    """

    def __init__(self, keys=['img', 'mask'], collect_keys=[], cfg=None):
        self.keys = keys
        self.collect_keys = collect_keys

    def __call__(self, sample):
        data = {}
        if len(sample['img'].shape) < 3:
            sample['img'] = np.expand_dims(sample['img'], -1)
        for key in sample.keys():
            if key in self.keys:
                data[key] = to_tensor(sample[key])
            if key in self.collect_keys:
                data[key] = sample[key]
        data['img'] = data['img'].permute(2, 0, 1)
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
    
class RandomLROffsetLABEL(object):
    def __init__(self,max_offset, cfg=None):
        self.max_offset = max_offset
    def __call__(self, sample):
        img = sample['img'] 
        label = sample['mask'] 
        offset = np.random.randint(-self.max_offset,self.max_offset)
        h, w = img.shape[:2]

        img = np.array(img)
        if offset > 0:
            img[:,offset:,:] = img[:,0:w-offset,:]
            img[:,:offset,:] = 0
        if offset < 0:
            real_offset = -offset
            img[:,0:w-real_offset,:] = img[:,real_offset:,:]
            img[:,w-real_offset:,:] = 0

        label = np.array(label)
        if offset > 0:
            label[:,offset:] = label[:,0:w-offset]
            label[:,:offset] = 0
        if offset < 0:
            offset = -offset
            label[:,0:w-offset] = label[:,offset:]
            label[:,w-offset:] = 0
        sample['img'] = img
        sample['mask'] = label
        
        return sample 

class RandomUDoffsetLABEL(object):
    def __init__(self,max_offset, cfg=None):
        self.max_offset = max_offset
    def __call__(self, sample):
        img = sample['img'] 
        label = sample['mask'] 
        offset = np.random.randint(-self.max_offset,self.max_offset)
        h, w = img.shape[:2]

        img = np.array(img)
        if offset > 0:
            img[offset:,:,:] = img[0:h-offset,:,:]
            img[:offset,:,:] = 0
        if offset < 0:
            real_offset = -offset
            img[0:h-real_offset,:,:] = img[real_offset:,:,:]
            img[h-real_offset:,:,:] = 0

        label = np.array(label)
        if offset > 0:
            label[offset:,:] = label[0:h-offset,:]
            label[:offset,:] = 0
        if offset < 0:
            offset = -offset
            label[0:h-offset,:] = label[offset:,:]
            label[h-offset:,:] = 0
        sample['img'] = img
        sample['mask'] = label
        return sample 
    
class Resize(object):
    def __init__(self, size, cfg=None):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, sample):
        out = list()
        sample['img'] = cv2.resize(sample['img'], self.size,
                              interpolation=cv2.INTER_CUBIC)
        if 'mask' in sample:
            sample['mask'] = cv2.resize(sample['mask'], self.size,
                                  interpolation=cv2.INTER_NEAREST)
        return sample
    
class RandomCrop(object):
    def __init__(self, size, cfg=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        h, w = img_group[0].shape[0:2]
        th, tw = self.size

        out_images = list()
        h1 = random.randint(0, max(0, h - th))
        w1 = random.randint(0, max(0, w - tw))
        h2 = min(h1 + th, h)
        w2 = min(w1 + tw, w)

        for img in img_group:
            assert (img.shape[0] == h and img.shape[1] == w)
            out_images.append(img[h1:h2, w1:w2, ...])
        return out_images

class CenterCrop(object):
    def __init__(self, size, cfg=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        h, w = img_group[0].shape[0:2]
        th, tw = self.size

        out_images = list()
        h1 = max(0, int((h - th) / 2))
        w1 = max(0, int((w - tw) / 2))
        h2 = min(h1 + th, h)
        w2 = min(w1 + tw, w)

        for img in img_group:
            assert (img.shape[0] == h and img.shape[1] == w)
            out_images.append(img[h1:h2, w1:w2, ...])
        return out_images
    
class RandomRotation(object):
    def __init__(self, degree=(-10, 10), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST), padding=None, cfg=None):
        self.degree = degree
        self.interpolation = interpolation
        self.padding = padding
        if self.padding is None:
            self.padding = [0, 0]

    def _rotate_img(self, sample, map_matrix):
        h, w = sample['img'].shape[0:2]
        sample['img'] = cv2.warpAffine(
            sample['img'], map_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)

    def _rotate_mask(self, sample, map_matrix):
        if 'mask' not in sample:
            return
        h, w = sample['mask'].shape[0:2]
        sample['mask'] = cv2.warpAffine(
            sample['mask'], map_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)


    def __call__(self, sample):
        v = random.random()
        if v < 0.5:
            degree = random.uniform(self.degree[0], self.degree[1])
            h, w = sample['img'].shape[0:2]
            center = (w / 2, h / 2)
            map_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
            self._rotate_img(sample, map_matrix)
            self._rotate_mask(sample, map_matrix)
        return sample

class RandomBlur(object):
    def __init__(self, applied, cfg=None):
        self.applied = applied

    def __call__(self, img_group):
        assert (len(self.applied) == len(img_group))
        v = random.random()
        if v < 0.5:
            out_images = []
            for img, a in zip(img_group, self.applied):
                if a:
                    img = cv2.GaussianBlur(
                        img, (5, 5), random.uniform(1e-6, 0.6))
                out_images.append(img)
                if len(img.shape) > len(out_images[-1].shape):
                    out_images[-1] = out_images[-1][...,
                                                    np.newaxis]  # single channel image
            return out_images
        else:
            return img_group
        
class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy Image with a probability of 0.5
    """

    def __init__(self, cfg=None):
        pass

    def __call__(self, sample):
        v = random.random()
        if v < 0.5:
            sample['img'] = np.fliplr(sample['img'])
            if 'mask' in sample: sample['mask'] = np.fliplr(sample['mask'])
            if 'lanes' in sample:
                width = sample['img'].shape[1]
                lanes = sample['lanes']
                new_lanes = []
                for lane in lanes:
                    new_lane = []
                    for p in lane:
                        new_lane.append(((width - 1) - p[0], p[1]))
                    new_lanes.append(new_lane)

        return sample

class Normalize(object):
    def __init__(self, img_norm, cfg=None):
        self.mean = np.array(img_norm['mean'], dtype=np.float32)
        self.std = np.array(img_norm['std'], dtype=np.float32)

    def __call__(self, sample):
        m = self.mean
        s = self.std
        img = sample['img'] 
        if len(m) == 1:
            img = img - np.array(m)  # single channel image
            img = img / np.array(s)
        else:
            img = img - np.array(m)[np.newaxis, np.newaxis, ...]
            img = img / np.array(s)[np.newaxis, np.newaxis, ...]
        sample['img'] = img

        return sample 


class Preprocess(object):
    def __init__(self,preprocess):
        self.preprocess = preprocess
    
    def __call__(self, data):
        for t in self.preprocess:
            data = t(data)
            if data is None:
                return None
        return data

class RandomAffine:
    def __init__(self, affine_ratio, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0),keys=[]):
        assert 0 <= affine_ratio <= 1
        self.affine_ratio = affine_ratio
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border
        self.key_list = keys

    def _transform_data(self, results, M, width, height):
        # transform img
        img = results["img"].copy()
        if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                im = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(0, 0, 0))
            else:  # affine
                im = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(0, 0, 0))
        results["img"] = im

        # transform lane
        for key in self.key_list:
            lanes = results[key].copy()
            new_lanes = []
            for lane in lanes:
                new_lane = []
                for p in lane:
                    p = np.expand_dims(np.array(p), axis=1)    # (2, 1)
                    p = np.concatenate((p, np.ones(shape=(1, 1), dtype=np.float)), axis=0)    # (3, 1)
                    new_p = (M[:2] @ p).squeeze().tolist()
                    new_lane.append(new_p)
                new_lanes.append(new_lane)
            results[key] = new_lanes

        # transform seg
        for key in results.get('seg_fields', []):
            seg = results[key].copy()
            if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
                if self.perspective:
                    seg = cv2.warpPerspective(seg, M, dsize=(width, height), flags=cv2.INTER_NEAREST, borderValue=0)
                else:  # affine
                    seg = cv2.warpAffine(seg, M[:2], dsize=(width, height), flags=cv2.INTER_NEAREST, borderValue=0)
                results[key] = seg

    def __call__(self, results):

        if random.random() < self.affine_ratio:
            img = results['img']
            height = img.shape[0] + self.border[0] * 2
            width = img.shape[1] + self.border[1] * 2

            # Center
            C = np.eye(3)
            C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
            C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

            # Perspective
            P = np.eye(3)
            P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
            P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

            # Rotation and Scale
            R = np.eye(3)
            a = random.uniform(-self.degrees, self.degrees)
            # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
            s = random.uniform(1 - self.scale, 1 + self.scale)
            # s = 2 ** random.uniform(-scale, scale)
            R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

            # Shear
            S = np.eye(3)
            S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
            S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

            # Translation
            T = np.eye(3)
            T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width  # x translation (pixels)
            T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height  # y translation (pixels)

            # Combined rotation matrix
            M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
            self._transform_data(results, M, width, height)


        return results
