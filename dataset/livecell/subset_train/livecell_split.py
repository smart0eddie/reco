from argparse import ArgumentParser
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
from pycocotools.coco import COCO

def write_data(path, data):
    with open(path, 'w') as f:
        for d in data:
            f.write("%s\n" % d)

def get_unsup(all_list, sup_list):
    return [x for x in all_list if x not in sup_list]

def get_file_name(coco: COCO, ids):
    imgs = coco.loadImgs(ids)
    return [os.path.splitext(img["file_name"])[0] for img in imgs]

root = "dataset/livecell/"

split_ann_files = [
    ("0_train2percent.json", 50),
    ("1_train4percent.json", 25),
    ("2_train5percent.json", 20),
    ("3_train25percent.json", 4),
    ("4_train50percent.json", 2),
]

parser = ArgumentParser(description="Semi-supervised segmentation")
parser.add_argument("--data_root", type=str, default="../Data/livecell-dataset/",
                        help="Root dir of datasets") 
args = parser.parse_args()

train_set_path = args.data_root + "livecell-dataset/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json"
val_set_path = args.data_root + "livecell-dataset/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json"
test_set_path = args.data_root + "livecell-dataset/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_test.json"

# train split
coco_full = COCO(train_set_path)
imgIds_full = coco_full.getImgIds()
train = get_file_name(coco_full, imgIds_full) 
write_data(root + "train.txt", train)

for split in split_ann_files:
    f, r = split
    ann_path = args.data_root + "livecell-dataset/LIVECell_dataset_2021/annotations/LIVECell_dataset_size_split/" + f
    
    coco = COCO(ann_path)
    sup_ids = coco.getImgIds()    
    unsup_ids = get_unsup(imgIds_full, sup_ids)
    
    sup = get_file_name(coco_full, sup_ids)    
    unsup = get_file_name(coco_full, unsup_ids)  
    
    print(sup)
    print(unsup)
    
    write_data(root + f"subset_train/train_labeled_1-{r}.txt", sup)
    write_data(root + f"subset_train/train_unlabeled_1-{r}.txt", unsup)
    
# val
coco_val = COCO(val_set_path)
val = get_file_name(coco_val, coco_val.getImgIds()) 
write_data(root + "val.txt", val)

# test
coco_test = COCO(val_set_path)
test = get_file_name(coco_test, coco_test.getImgIds())
write_data(root + "test.txt", test)