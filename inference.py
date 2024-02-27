# -*- coding: utf-8 -*-
import os
import time
import copy
import torch
import random
import argparse
import pandas as pd
from skimage import io, transform
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import torch.nn as nn
from tqdm import tqdm

from model.Merge_CNN_model import CNNMerge
from model.TranSalNet_ViT_multidecoder import TranSalNet
from utils.visualization import visualization


def init_models(args):
    path_to_ViT_multidecoder = "./weights/ViT_multidecoder.pth"
    path_to_СNNMerge = "./weights/CNNMerge.pth"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_ViT_multidecoder = TranSalNet()
    model_ViT_multidecoder = model_ViT_multidecoder.to(device)
    model_ViT_multidecoder.load_state_dict(torch.load(path_to_ViT_multidecoder))
    model_ViT_multidecoder.eval()
    
    model_СNNMerge = CNNMerge()
    model_СNNMerge = model_СNNMerge.to(device)
    model_СNNMerge.load_state_dict(torch.load(path_to_СNNMerge))
    model_СNNMerge.eval()
    
    return model_ViT_multidecoder, model_СNNMerge
    

def main(args):
    model_ViT_multidecoder, model_СNNMerge = init_models(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    start = time.time()
    
    # transform image
    img_path = args.img_path
    name_img = img_path.split("/")[-1].split(".")[0]
    img = Image.open(img_path).convert("RGB")
    img = np.array(img) / 255.
    shape_img = img.shape
    shape_img_w = shape_img[0]
    shape_img_h = shape_img[1]
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)
    img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = TF.resize(img, (288, 384))
    img = img.type(torch.float32).to(device)
       
    # predict saliency
    pred_map_1, pred_map_2 = model_ViT_multidecoder(img.unsqueeze(0))
    pred_map = model_СNNMerge(pred_map_1, pred_map_2)
    
    pred_map_1 = TF.resize(pred_map_1, (shape_img_w, shape_img_h))
    pred_map_2 = TF.resize(pred_map_2, (shape_img_w, shape_img_h))
    pred_map = TF.resize(pred_map, (shape_img_w, shape_img_h))
    
    # save maps to result
    path_to_save = "./results/" + name_img
    os.mkdir(path_to_save)
    save_image(pred_map_1, path_to_save + "/map_decoder_1.png")
    save_image(pred_map_2, path_to_save + "/map_decoder_2.png")
    save_image(pred_map, path_to_save + "/map.png")
    
    if args.color:
        visualization(img_path, path_to_save + "/map.png", path_to_save + "/color_map.png")
    
    print("Total time: ", time.time()-start)
    print("The images are saved to the results folder")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default="./inputs/img_1.jpg", type=str, help="path to image") 
    parser.add_argument("--color", default=False, type=str, help="save color map on the image") 
    args = parser.parse_args()
    main(args)