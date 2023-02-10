import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np

import utils.utils

utils.utils.set_random_seed(0)

def write_data(path, data):
    with open(path, 'w') as f:
        for d in data:
            f.write("%s\n" % d)

root = "dataset/crag/"


# data
testset = [
    f"test_{i+1}" for i in range(40)
]

trainset = [
   f"train_{i+1}" for i in range(173)
]

trainset_perm = np.random.permutation(trainset)
print(trainset_perm)

# save split
write_data(root + "val.txt", testset)
write_data(root + "train.txt", trainset)

ratio = [2, 4, 8, 16]
for r in ratio:
    sup = []
    unsup = []

    for i, v in enumerate(trainset_perm):
        if (r - 1) == (i % r):
            sup.append(v)
        else:
            unsup.append(v)  

    print(sup)
    print(unsup)

    write_data(root + f"subset_train/train_labeled_1-{r}.txt", sup)
    write_data(root + f"subset_train/train_unlabeled_1-{r}.txt", unsup)