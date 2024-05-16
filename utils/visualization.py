import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def find_th(n, max_item, min_item):
    th_list = []
    #max_item = 230
    #min_item = 110
    for i in range(n):
        th_list.append(int((max_item - min_item) / (i + 1) + min_item))
    th_list.append(min_item)
    return th_list


def make_pil(mask, th_list):
    true_1 = (np.array(mask >= th_list[0]) * np.ones(mask.shape) * 255).astype(np.uint8)
    true_2 = (np.array(mask >= th_list[1]) * np.ones(mask.shape) * 255).astype(np.uint8)
    true_2 = true_2 - true_1
    true_3 = (np.array(mask >= th_list[2]) * np.ones(mask.shape) * 255).astype(np.uint8)
    true_3 = true_3 - true_2 - true_1
    true_4 = (np.array(mask >= th_list[3]) * np.ones(mask.shape) * 255).astype(np.uint8)
    true_4 = true_4 - true_3 - true_2 - true_1
    
    # make true_1 red
    true_1[:,:,1] = np.zeros((true_1.shape[0], true_1.shape[1]))
    true_1[:,:,2] = np.zeros((true_1.shape[0], true_1.shape[1]))
    # make true_2 orange
    true_2[:,:,1] = (true_2[:,:,1]/2).astype(np.uint8)
    true_2[:,:,2] = np.zeros((true_2.shape[0], true_2.shape[1]))
    # make true_3 yellow
    true_3[:,:,2] = np.zeros((true_3.shape[0], true_3.shape[1]))
    # make true_3 green
    true_4[:,:,0] = np.zeros((true_4.shape[0], true_4.shape[1]))
    true_4[:,:,2] = np.zeros((true_4.shape[0], true_4.shape[1]))
    
    final_true = true_1 + true_2 + true_3 + true_4
    final_true_PIL = Image.fromarray(final_true)
    
    return final_true_PIL


def visualization(img_path, map_path, out_path, n=3, max_item=230, min_item=110):
    image_PIL = Image.open(img_path).convert("RGB")
    image = np.array(image_PIL)
    
    mask_PIL = Image.open(map_path).convert("RGB")
    mask = np.array(mask_PIL)
    
    th_list = find_th(n, max_item, min_item)
    final_true_PIL = make_pil(mask, th_list)
    result_PIL = Image.blend(image_PIL, final_true_PIL, alpha=0.4)
    result_PIL.save(out_path) 
    