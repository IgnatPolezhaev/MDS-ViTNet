import os
import cv2
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from utils.loss_function import SaliencyLoss, AUC

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

    def __init__(self, p, shape_r=288, shape_c=384, iftrain=False):
        self.p = p
        self.shape_r = shape_r
        self.shape_c = shape_c
        self.iftrain = iftrain
    
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

    
class MyTransformCNNMerge:

    def __init__(self, shape_r=288, shape_c=384):
        self.shape_r = shape_r
        self.shape_c = shape_c

    def __call__(self, map1, map2, smap):
        
        #resize
        map1 = TF.resize(map1, (self.shape_r, self.shape_c))
        map2 = TF.resize(map2, (self.shape_r, self.shape_c))
        smap = TF.resize(smap, (self.shape_r, self.shape_c))

        return map1, map2, smap


class MyDataset(Dataset):

    def __init__(self, ids, stimuli_dir, saliency_dir, transform=None):
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

        sample = {'image': img, 'saliency': smap, 'path': im_path}

        return sample
    
    
class MyDatasetCNNMerge(Dataset):

    def __init__(self, ids, map1_dir, map2_dir, saliency_dir, transform=None):
        self.ids = ids
        self.map1_dir = map1_dir
        self.map2_dir = map2_dir
        self.saliency_dir = saliency_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        map1_path = self.map1_dir + self.ids.iloc[idx, 0]
        map1 = Image.open(map1_path).convert('L')
        smap1 = np.expand_dims(np.array(map1) / 255., axis=0)
        smap1 = torch.from_numpy(smap1)
        
        map2_path = self.map2_dir + self.ids.iloc[idx, 0]
        map2 = Image.open(map2_path).convert('L')
        smap2 = np.expand_dims(np.array(map2) / 255., axis=0)
        smap2 = torch.from_numpy(smap2)

        smap_path = self.saliency_dir + self.ids.iloc[idx, 1]
        saliency = Image.open(smap_path).convert('L')
        smap = np.expand_dims(np.array(saliency) / 255., axis=0)
        smap = torch.from_numpy(smap)
        
        if self.transform:
            smap1, smap2, smap = self.transform(smap1, smap2, smap)

        sample = {'map1': smap1, 'map2': smap2, 'saliency': smap}

        return sample
    
    
def compute_metric(model, dataloader, device, flag, t=10):
    loss_fn = SaliencyLoss()

    if flag == 3:
        val_loss = 0.0
        val_loss_cc = [0.0, 0.0]
        val_loss_sim = [0.0, 0.0]
        val_loss_kldiv = [0.0, 0.0]
        val_loss_nss = [0.0, 0.0]
        val_auc = [0.0, 0.0]
    else:
        val_loss = 0.0
        val_loss_cc = 0.0
        val_loss_sim = 0.0
        val_loss_kldiv = 0.0
        val_loss_nss = 0.0
        val_auc = 0.0

    for i_batch, sample_batched in tqdm(enumerate(dataloader)):
        stimuli, smap = sample_batched['image'], sample_batched['saliency']
        stimuli, smap = stimuli.type(torch.float32), smap.type(torch.float32)
        stimuli, smap = stimuli.to(device), smap.to(device)

        with torch.no_grad():
            if flag == 3:
                outputs_1, outputs_2 = model(stimuli)
            else:
                outputs = model(stimuli)

            b_s = stimuli.shape[0]
            if flag == 3:
                batch_auc_1 = 0.0
                batch_auc_2 = 0.0
            else:
                batch_auc = 0.0
            for i in range(b_s):
                img_true = (smap[i].squeeze().cpu().detach().numpy()*255).astype(int)
                if flag == 3:
                    img_pred_1 = (outputs_1[i].squeeze().cpu().detach().numpy()*255).astype(int)
                    img_pred_2 = (outputs_2[i].squeeze().cpu().detach().numpy()*255).astype(int)
                    batch_auc_1 += AUC(img_true, img_pred_1, t=t)
                    batch_auc_2 += AUC(img_true, img_pred_2, t=t)
                else:
                    img_pred = (outputs[i].squeeze().cpu().detach().numpy()*255).astype(int)
                    batch_auc += AUC(img_true, img_pred, t=t)

            if flag == 3:
                batch_auc_1 = batch_auc_1 / b_s
                batch_auc_2 = batch_auc_2 / b_s
            else:
                batch_auc = batch_auc / b_s

            if flag == 3:
                loss_1 = -2*loss_fn(outputs_1, smap, loss_type='cc')
                loss_1 = loss_1 - loss_fn(outputs_1, smap, loss_type='sim')
                loss_1 = loss_1 + 10*loss_fn(outputs_1, smap, loss_type='kldiv')
                #loss_1 = loss_1 - loss_fn(outputs_1, smap, loss_type='nss')
                
                loss_2 = -2*loss_fn(outputs_2, smap, loss_type='cc')
                loss_2 = loss_2 - loss_fn(outputs_2, smap, loss_type='sim')
                loss_2 = loss_2 + 10*loss_fn(outputs_2, smap, loss_type='kldiv')
                #loss_2 = loss_2 - loss_fn(outputs_2, smap, loss_type='nss')
                
                loss = loss_1 + loss_2
            else:
                loss = -2*loss_fn(outputs, smap, loss_type='cc')
                loss = loss - loss_fn(outputs, smap, loss_type='sim')
                loss = loss + 10*loss_fn(outputs, smap, loss_type='kldiv')
                #loss = loss - loss_fn(outputs, smap, loss_type='nss')

            if flag == 3:
                val_loss += loss.item()
                val_loss_cc[0] += loss_fn(outputs_1, smap, loss_type='cc').item()
                val_loss_cc[1] += loss_fn(outputs_2, smap, loss_type='cc').item()
                val_loss_sim[0] += loss_fn(outputs_1, smap, loss_type='sim').item()
                val_loss_sim[1] += loss_fn(outputs_2, smap, loss_type='sim').item()
                val_loss_kldiv[0] += loss_fn(outputs_1, smap, loss_type='kldiv').item()
                val_loss_kldiv[1] += loss_fn(outputs_2, smap, loss_type='kldiv').item()
                val_loss_nss[0] += loss_fn(outputs_1, smap, loss_type='nss').item()
                val_loss_nss[1] += loss_fn(outputs_2, smap, loss_type='nss').item()
                val_auc[0] += batch_auc_1
                val_auc[1] += batch_auc_2
            else:
                val_loss += loss.item()
                val_loss_cc += loss_fn(outputs, smap, loss_type='cc').item()
                val_loss_sim += loss_fn(outputs, smap, loss_type='sim').item()
                val_loss_kldiv += loss_fn(outputs, smap, loss_type='kldiv').item()
                val_loss_nss += loss_fn(outputs, smap, loss_type='nss').item()
                val_auc += batch_auc

    if flag == 3:
        val_loss = val_loss / len(dataloader)
        val_loss_cc[0] = val_loss_cc[0] / len(dataloader)
        val_loss_cc[1] = val_loss_cc[1] / len(dataloader)
        val_loss_sim[0] = val_loss_sim[0] / len(dataloader)
        val_loss_sim[1] = val_loss_sim[1] / len(dataloader)
        val_loss_kldiv[0] = val_loss_kldiv[0] / len(dataloader)
        val_loss_kldiv[1] = val_loss_kldiv[1] / len(dataloader)
        val_loss_nss[0] = val_loss_nss[0] / len(dataloader)
        val_loss_nss[1] = val_loss_nss[1] / len(dataloader)
        val_auc[0] = val_auc[0] / len(dataloader)
        val_auc[1] = val_auc[1] / len(dataloader)
    else:
        val_loss = val_loss / len(dataloader)
        val_loss_cc = val_loss_cc / len(dataloader)
        val_loss_sim = val_loss_sim / len(dataloader)
        val_loss_kldiv = val_loss_kldiv / len(dataloader)
        val_loss_nss = val_loss_nss / len(dataloader)
        val_auc = val_auc / len(dataloader)

    return val_loss, val_loss_cc, val_loss_sim, val_loss_kldiv, val_loss_nss, val_auc


def compute_metric_CNNMerge(model, dataloader, device, t=10):
    loss_fn = SaliencyLoss()

    val_loss = 0.0
    val_loss_cc = 0.0
    val_loss_sim = 0.0
    val_loss_kldiv = 0.0
    val_loss_nss = 0.0
    val_auc = 0.0

    for i_batch, sample_batched in tqdm(enumerate(dataloader)):
        smap1, smap2, smap = sample_batched['map1'], sample_batched['map2'], sample_batched['saliency']
        smap1, smap2, smap = smap1.type(torch.float32), smap2.type(torch.float32), smap.type(torch.float32)
        smap1, smap2, smap = smap1.to(device), smap2.to(device), smap.to(device)

        with torch.no_grad():
            outputs = model(smap1, smap2)

            b_s = smap.shape[0]
            batch_auc = 0.0
            
            for i in range(b_s):
                img_true = (smap[i].squeeze().cpu().detach().numpy()*255).astype(int)
                img_pred = (outputs[i].squeeze().cpu().detach().numpy()*255).astype(int)
                batch_auc += AUC(img_true, img_pred, t=t)

            batch_auc = batch_auc / b_s

            loss = -2*loss_fn(outputs, smap, loss_type='cc')
            loss = loss - loss_fn(outputs, smap, loss_type='sim')
            loss = loss + 10*loss_fn(outputs, smap, loss_type='kldiv')

            val_loss += loss.item()
            val_loss_cc += loss_fn(outputs, smap, loss_type='cc').item()
            val_loss_sim += loss_fn(outputs, smap, loss_type='sim').item()
            val_loss_kldiv += loss_fn(outputs, smap, loss_type='kldiv').item()
            val_loss_nss += loss_fn(outputs, smap, loss_type='nss').item()
            val_auc += batch_auc

    val_loss = val_loss / len(dataloader)
    val_loss_cc = val_loss_cc / len(dataloader)
    val_loss_sim = val_loss_sim / len(dataloader)
    val_loss_kldiv = val_loss_kldiv / len(dataloader)
    val_loss_nss = val_loss_nss / len(dataloader)
    val_auc = val_auc / len(dataloader)

    return val_loss, val_loss_cc, val_loss_sim, val_loss_kldiv, val_loss_nss, val_auc
