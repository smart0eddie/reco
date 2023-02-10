from argparse import ArgumentParser
import glob
import os, os.path

import cv2
import numpy as np
from tqdm import tqdm

def ensure_crop_boundary(image_size, crop_position, crop_size):
    """_summary_

    Args:
        image_size (_type_): 
        crop_position (_type_):
        crop_size (_type_): 
    """
    
    if crop_position > image_size - crop_size:
        crop_position = image_size - crop_size
       
    return crop_position

def generate_crops(crop_dir, image_name, image: np.ndarray, image_size, crop_size, crop_stride, nzthr=0.01):
    ih, iw = image_size[0], image_size[1]    
    ch, cw = crop_size[0], crop_size[1]
    sh, sw = crop_stride[0], crop_stride[1]
    
    if ch > ih:
        ch = ih
    if cw > iw:
        cw = iw
    
    y_last = ih - ch
    x_last = iw - cw  
    nza = ch * cw * nzthr 
    
    for cy in range(0, ih+sh, sh):        
        if cy > y_last:
            cy = y_last
            
        for cx in range(0, iw+sw, sw):
            if cx > x_last:
                cx = x_last
            
            crop = image[cy:cy+ch, cx:cx+cw]
            tmp = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if (len(crop.shape) == 3 and crop.shape[2] == 3) else crop
            if cv2.countNonZero(tmp) >= nza:
                crop_path = crop_dir + image_name + f"_{cx}_{cy}.png"
                cv2.imwrite(crop_path, crop)
                
            if cx == x_last:
                break
            
        if cy == y_last:
            break

def process_dataset(root, crop_size, crop_stride, nzthr=0.01):
    img_dir = root + "Img/"
    img_crop_dir = root + "Img_crop/"
    mask_dir = root + "Truth/"
    mask_crop_dir = root + "Truth_crop/"
    
    os.makedirs(img_crop_dir, exist_ok=True)
    os.makedirs(mask_crop_dir, exist_ok=True)
    
    file_list = glob.glob(img_dir + "*.png")
    for file_path in tqdm(file_list):        
        file_name = os.path.splitext(os.path.basename(file_path))[0] 
        
        img = cv2.imread(file_path)
        h, w, c = img.shape         
        generate_crops(img_crop_dir, file_name, img, (h, w), crop_size, crop_stride)             
        
        mask_name = "Mask_" + file_name
        mask_path = mask_dir + mask_name + ".png"
        mask = cv2.imread(mask_path)
        generate_crops(mask_crop_dir, mask_name, mask, (h, w), crop_size, crop_stride)

parser = ArgumentParser(description="Semi-supervised segmentation")
parser.add_argument("--data_root", type=str, default="../Data/MTCHI/",
                        help="Root dir of datasets") 
args = parser.parse_args()

train_set_path = args.data_root + "MTCHI/MTCHI Dataset/MTCHI Dataset/Task2/Training/"

crop_size = (1024, 1024)
crop_stride = (512, 512)
nzthr = 0.01

process_dataset(train_set_path, crop_size, crop_stride, nzthr)