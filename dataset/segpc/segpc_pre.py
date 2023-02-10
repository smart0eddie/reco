from argparse import ArgumentParser
import glob
import os, os.path

import cv2
import numpy as np
from tqdm import tqdm

def process_mask(mask: np.ndarray):
    return (mask / 20).astype(np.uint8)

def merge_mask(file_list, img_size):
    """_summary_

    Args:
        file_lise (_type_): _description_
        img_size (_type_): (h, w, c)
    """
    
    mask = np.zeros(img_size, np.uint8)
    
    for path in file_list:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask |= process_mask(m)
        
    return mask

def process_dataset(root):
    img_dir = root + "x/"
    mask_dir = root + "y/"
    merged_mask_dir = root + "y_merge/"
    os.makedirs(merged_mask_dir, exist_ok=True)
    
    file_list = glob.glob(img_dir + "*.bmp")
    for file_path in tqdm(file_list):        
        file_name = os.path.splitext(os.path.basename(file_path))[0] 
        
        img = cv2.imread(file_path)
        h, w, c = img.shape              
        
        mask_list = glob.glob(mask_dir + file_name + "_*.bmp")
        
        mask = merge_mask(mask_list, (h, w))
        merge_path = merged_mask_dir + file_name + ".png"
        cv2.imwrite(merge_path, mask)

parser = ArgumentParser(description="Semi-supervised segmentation")
parser.add_argument("--data_root", type=str, default="../Data/SegPC2021/",
                        help="Root dir of datasets") 
args = parser.parse_args()

train_set_path = args.data_root + "SegPC2021/TCIA_SegPC_dataset/train/train/train/"
val_set_path = args.data_root + "SegPC2021/TCIA_SegPC_dataset/validation/validation/"

process_dataset(train_set_path)
process_dataset(val_set_path)