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

root = "dataset/monuseg/"


# data
testset = [
    "TCGA-2Z-A9J9-01A-01-TS1",
    "TCGA-44-2665-01B-06-BS6",
    "TCGA-69-7764-01A-01-TS1",
    "TCGA-A6-6782-01A-01-BS1",
    "TCGA-AC-A2FO-01A-01-TS1",
    "TCGA-AO-A0J2-01A-01-BSA",
    "TCGA-CU-A0YN-01A-02-BSB",
    "TCGA-EJ-A46H-01A-03-TSC",
    "TCGA-FG-A4MU-01B-01-TS1",
    "TCGA-GL-6846-01A-01-BS1",
    "TCGA-HC-7209-01A-01-TS1",
    "TCGA-HT-8564-01Z-00-DX1",
    "TCGA-IZ-8196-01A-01-BS1",
    "TCGA-ZF-A9R5-01A-01-TS1",
]

trainset = [
   "TCGA-18-5592-01Z-00-DX1",
    "TCGA-21-5784-01Z-00-DX1",
    "TCGA-21-5786-01Z-00-DX1",
    "TCGA-38-6178-01Z-00-DX1",
    "TCGA-49-4488-01Z-00-DX1",
    "TCGA-50-5931-01Z-00-DX1",
    "TCGA-A7-A13E-01Z-00-DX1",
    "TCGA-A7-A13F-01Z-00-DX1",
    "TCGA-AR-A1AK-01Z-00-DX1",
    "TCGA-AR-A1AS-01Z-00-DX1",
    "TCGA-AY-A8YK-01A-01-TS1",
    "TCGA-B0-5698-01Z-00-DX1",
    "TCGA-B0-5710-01Z-00-DX1",
    "TCGA-B0-5711-01Z-00-DX1",
    "TCGA-BC-A217-01Z-00-DX1",
    "TCGA-CH-5767-01Z-00-DX1",
    "TCGA-DK-A2I6-01A-01-TS1",
    "TCGA-E2-A1B5-01Z-00-DX1",
    "TCGA-E2-A14V-01Z-00-DX1",
    "TCGA-F9-A8NY-01Z-00-DX1",
    "TCGA-FG-A87N-01Z-00-DX1",
    "TCGA-G2-A2EK-01A-02-TSB",
    "TCGA-G9-6336-01Z-00-DX1",
    "TCGA-G9-6348-01Z-00-DX1",
    "TCGA-G9-6356-01Z-00-DX1",
    "TCGA-G9-6362-01Z-00-DX1",
    "TCGA-G9-6363-01Z-00-DX1",
    "TCGA-HE-7128-01Z-00-DX1",
    "TCGA-HE-7129-01Z-00-DX1",
    "TCGA-HE-7130-01Z-00-DX1",
    "TCGA-KB-A93J-01A-01-TS1",
    "TCGA-MH-A561-01Z-00-DX1",
    "TCGA-NH-A8F7-01A-01-TS1",
    "TCGA-RD-A8N9-01A-01-TS1",
    "TCGA-UZ-A9PJ-01Z-00-DX1",
    "TCGA-UZ-A9PN-01Z-00-DX1",
    "TCGA-XS-A8TJ-01Z-00-DX1",
]

train_num = len(trainset)
trainset_perm = np.random.permutation(trainset)
print(trainset_perm)

# save split
write_data(root + "val.txt", testset)
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