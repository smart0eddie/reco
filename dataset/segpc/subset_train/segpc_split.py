from argparse import ArgumentParser
import glob
import os, os.path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np

import utils.utils

utils.utils.set_random_seed(0)

def write_data(path, data):
    with open(path, 'w') as f:
        for d in data:
            f.write("%s\n" % d)

def get_file_names(root):
    img_dir = root + "x/"
    file_list = glob.glob(img_dir + "*.bmp")
    
    return [os.path.splitext(os.path.basename(file_path))[0] for file_path in file_list]

root = "dataset/segpc/"

parser = ArgumentParser(description="Semi-supervised segmentation")
parser.add_argument("--data_root", type=str, default="../Data/SegPC2021/",
                        help="Root dir of datasets") 
args = parser.parse_args()

train_set_path = args.data_root + "SegPC2021/TCIA_SegPC_dataset/train/train/train/"
val_set_path = args.data_root + "SegPC2021/TCIA_SegPC_dataset/validation/validation/"

valset = get_file_names(val_set_path)
trainset = get_file_names(train_set_path)

train_num = len(trainset)
trainset_perm = np.random.permutation(trainset)
print(trainset_perm)

# save split
write_data(root + "val.txt", valset)
write_data(root + "train.txt", trainset)

ratio = [2, 4, 8, 16]
for r in ratio:
    sup_num = train_num // r
    
    sup = trainset_perm[:sup_num]
    unsup = trainset_perm[sup_num:]    

    print(sup)
    print(unsup)

    write_data(root + f"subset_train/train_labeled_1-{r}.txt", sup)
    write_data(root + f"subset_train/train_unlabeled_1-{r}.txt", unsup)
