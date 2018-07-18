import os

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset


class SemSegDataset(Dataset):
    def __init__(self, dir_name='images', patch_size=(256, 256, 3), patch_sel='random', zero_mean=True):
        """
        Dataset for loading image batches. Assumes only one big image in dataset for now.
        """
        self.patch_size = patch_size
        self.patch_sel = patch_sel

        image_file = os.path.join(dir_name, 'rgb.png')
        self.image_data = io.imread(image_file)
        self.image_mask = self.image_data[:, :, -1]
        self.image_data = self.image_data[:, :, :3].astype(np.float32)

        gt_file = os.path.join(dir_name, 'gt.png')
        self.gt_data = io.imread(gt_file)
        self.gt_data = self.gt_data[:, :, 0].astype(np.float32) / 255
        self.gt_data = np.stack([self.gt_data, 1 - self.gt_data], axis=-1)
        if zero_mean:
            self.image_data = self.image_data - self.image_data.mean(axis=(0, 1))

        if self.patch_sel == 'random':
            self.total_patches = 1
        else:
            step_size = np.asarray(patch_size) * .8
            self.pidx = patch_idx(self.image_data.shape, patch_size, step_size.astype(np.int32))
            self.total_patches = len(self.pidx)

    def __len__(self):
        return self.total_patches

    def __getitem__(self, item):
        if self.patch_sel == 'random':  # ignore idx
            startx = np.random.randint(0, self.image_data.shape[0] - self.patch_size[0])
            starty = np.random.randint(0, self.image_data.shape[1] - self.patch_size[1])
            image_patch = self.image_data[startx:startx + self.patch_size[0], starty:starty + self.patch_size[1], :]
            gt_patch = self.gt_data[startx:startx + self.patch_size[0], starty:starty + self.patch_size[1], :]

        else:
            start_i, start_j, end_i, end_j = self.pidx[item]
            image_patch = self.image_data[start_i:end_i, start_j:end_j, :]
            gt_patch = self.gt_data[start_i:end_i, start_j:end_j, :]
        gt_patch = np.argmax(gt_patch, axis=-1)
        return {'image': torch.from_numpy(image_patch.transpose((2, 0, 1))),
                'gt': torch.from_numpy(gt_patch)}
        # 'gt': torch.from_numpy(gt_patch.transpose((2, 0, 1)))}


def patch_idx(image_shape, patch_size, step):
    x = list(range(0, image_shape[0] - patch_size[0], step[0]))
    y = list(range(0, image_shape[1] - patch_size[1], step[1]))
    x.append(image_shape[0] - patch_size[0])
    y.append(image_shape[1] - patch_size[1])
    xy = [(i, j, i + patch_size[0], j + patch_size[1]) for i in x for j in y]
    return xy


def predict_complete(fcn_model, image_data, patch_size=(256, 256, 3), step=(220, 220, 3)):
    image_shape = image_data.shape
    pidx = patch_idx(image_shape, patch_size, step)

    response_map = np.zeros(shape=image_shape[:2] + (2,))
    for start_i, start_j, end_i, end_j in pidx:
        img_patch = image_data[start_i:end_i, start_j:end_j]
        prob_out = fcn_model.predict_patch(img_patch)
        response_map[start_i:end_i, start_j:end_j] = np.maximum(response_map[start_i:end_i, start_j:end_j], prob_out)

    return response_map
