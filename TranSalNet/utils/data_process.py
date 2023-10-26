import cv2
from PIL import Image
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random


def preprocess_img(img_dir, shape_r=480, shape_c=640, channels=3):

    if channels == 1:
        img = cv2.imread(img_dir, 0)
    elif channels == 3:
        img = cv2.imread(img_dir)

    img_padded = np.ones((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)
    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c
    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:,
        ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))

        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows),
        :] = img

    return img_padded


def postprocess_img(pred, org_dir):
    pred = np.array(pred)
    org = cv2.imread(org_dir, 0)
    shape_r = org.shape[0]
    shape_c = org.shape[1]
    predictions_shape = pred.shape

    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    return img

class MyTransform:

    def __init__(self, p, angles, shape_r=288, shape_c=384, iftrain=False):
        self.p = p
        self.angles = angles
        self.shape_r = shape_r
        self.shape_c = shape_c
        self.iftrain = iftrain

    def rotate(self, image, map, angle):
        return TF.rotate(image, angle), TF.rotate(map, angle)
    
    def horizontal_flip(self, image, map):
        return TF.hflip(image), TF.hflip(map)
    
    def gaussian_blur(self, image, map):
        return TF.gaussian_blur(image, 3), map
    
    def adjust_brightness(self, image, map):
        brightness_factor = random.random() + 0.5
        return TF.adjust_brightness(image,brightness_factor), map
    
    def adjust_contrast(self, image, map):
        contrast_factor = random.random() + 0.5
        return TF.adjust_contrast(image, contrast_factor), map
    
    def adjust_saturation(self, image, map):
        saturation_factor = random.random() + 0.5
        return TF.adjust_saturation(image, saturation_factor), map
    
    def adjust_sharpness(self, image, map):
        sharpness_factor = random.random() + 0.5
        return TF.adjust_sharpness(image, sharpness_factor), map

    def __call__(self, image, map):
        angle = 0

        if self.iftrain:
            if random.random() < self.p:
                image, map = self.horizontal_flip(image, map)

            if random.random() < self.p:
                image, map = self.gaussian_blur(image, map)

            if random.random() < self.p:
                image, map = self.adjust_brightness(image, map)

            if random.random() < self.p:
                image, map = self.adjust_contrast(image, map)

            if random.random() < self.p:
                image, map = self.adjust_saturation(image, map)

            if random.random() < self.p:
                image, map = self.adjust_sharpness(image, map)

        #normalize
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        #resize
        image = TF.resize(image, (self.shape_r, self.shape_c))
        map = TF.resize(map, (self.shape_r, self.shape_c))

        return image, map


class MyDataset(Dataset):
    """Load dataset."""

    def __init__(self, ids, stimuli_dir, saliency_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ids = ids
        self.stimuli_dir = stimuli_dir
        self.saliency_dir = saliency_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_path = self.stimuli_dir + self.ids.iloc[idx, 0]
        image = Image.open(im_path).convert('RGB')
        img = np.array(image) / 255.
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        smap_path = self.saliency_dir + self.ids.iloc[idx, 1]
        saliency = Image.open(smap_path)
        smap = np.expand_dims(np.array(saliency) / 255., axis=0)
        smap = torch.from_numpy(smap)

        if self.transform:
            img, smap = self.transform(img, smap)

        sample = {'image': img, 'saliency': smap}

        return sample





