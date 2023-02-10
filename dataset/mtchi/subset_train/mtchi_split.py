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

def get_file_names(img_dir):
    file_list = glob.glob(img_dir + "*.png")
    
    return [os.path.splitext(os.path.basename(file_path))[0] for file_path in file_list]

root = "dataset/mtchi/"

parser = ArgumentParser(description="Semi-supervised segmentation")
parser.add_argument("--data_root", type=str, default="../Data/MTCHI/",
                        help="Root dir of datasets") 
args = parser.parse_args()

train_set_path = args.data_root + "MTCHI/MTCHI Dataset/MTCHI Dataset/Task2/Training/Img_crop/"
test_set_path = args.data_root + "MTCHI/MTCHI Dataset/MTCHI Dataset/Task2/Test/Img/"

testset = get_file_names(test_set_path)
trainset = get_file_names(train_set_path)

train_num = len(trainset)
trainset_perm = np.random.permutation(trainset)
print(trainset_perm)

# save split
write_data(root + "test.txt", testset)
write_data(root + "train_crop.txt", trainset)

ratio = [2, 4, 8, 16]
for r in ratio:
    sup_num = train_num // r
    
    sup = trainset_perm[:sup_num]
    unsup = trainset_perm[sup_num:]    

    print(sup)
    print(unsup)

    write_data(root + f"subset_train/train_crop_labeled_1-{r}.txt", sup)
    write_data(root + f"subset_train/train_crop_unlabeled_1-{r}.txt", unsup)