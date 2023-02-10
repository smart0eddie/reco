from argparse import ArgumentParser
import os
import string
from typing import Dict, List
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as COCOMask
import skimage

class ParseCOCOAnn(object):    
    def __init__(self) -> None:
        pass
    
    def parse_dataset_annotation(self, ann_path: string, mask_dir: string, color=1, binary=True):
        os.makedirs(mask_dir, exist_ok=True)
        
        coco = COCO(ann_path)
        
        imgIds = coco.getImgIds()
        imgs = coco.loadImgs(ids=imgIds)
        
        for img in tqdm(imgs):
            file_name = os.path.splitext(img["file_name"])[0]
            w, h = img["width"], img["height"]
                    
            annIds = coco.getAnnIds(imgIds=img['id'])
            anns = coco.loadAnns(annIds)
            
            mask = self.parse_image_annotation(anns, (h, w, 1), color=color, binary=binary)
                        
            mask_path = mask_dir + file_name + "_b.png"
            cv2.imwrite(mask_path, mask)
                   
        #     ## for debug                
        #     blank = np.zeros((h, w, 1), np.uint8)
        #     plt.imshow(blank)
        #     plt.show(block=False)
        #     coco.showAnns(anns)
                    
        #     cv2.imshow("mask", mask)
        #     cv2.waitKey(1)
        #     print(file_name)
        #     plt.clf()
        
        # print(imgIds)

    def parse_image_annotation(self, anns: List[Dict], img_size, color=1, binary=True):
        polygons = []
        
        mask = np.zeros(img_size, np.uint8)
        
        for ann in anns:
            if self.is_polygon(ann):
                pts = self.parse_polygon_annotation(ann["segmentation"][0])
            
                if len(ann["segmentation"]) > 1:
                    print(len(ann["segmentation"]) )
                
                parsed_ann = dict(pts=pts)
                polygons.append(parsed_ann)
            else:
                partial_mask = self.parse_rle_annotation(ann, img_size, color, binary)
                mask |= partial_mask
        
        poly_mask = self.render_polygon_mask(polygons, img_size, color, binary)
        mask |= poly_mask
        
        return mask

    def is_polygon(self, ann: Dict):
        """_summary_

        Args:
            ann (Dict): _description_
        """
        
        return type(ann['segmentation']) == list

    def parse_polygon_annotation(self, poly_pts):
        """
        Parse list of points from [x1, y1, ...] to [(x1, y1), ...]

        Args:
            pts_ann (_type_): _description_

        Returns:
            _type_: _description_
        """
        pts = []
        
        pt_iter = iter(poly_pts)
        for x, y in zip(pt_iter, pt_iter):
            xi = round(x)
            yi = round(y)
            
            pts.append((xi, yi))
            
        return pts

    def render_polygon_mask(self, anns, img_size, color=1, binary=True):
        """_summary_

        Args:
            anns (_type_): _description_
            img_size (_type_): (h, w, c)
        """

        mask = np.zeros(img_size, np.uint8)
        
        for idx, ann in enumerate(anns):
            c = color if binary else idx
            
            pts = [np.transpose(np.array([ann["pts"]]), (1, 0, 2))]
            mask = cv2.drawContours(mask, pts, contourIdx=-1, color=c, thickness=cv2.FILLED)

        return mask

    def parse_rle_annotation(self, rle_ann, image_size, color=1, binary=True):
        """_summary_

        Args:
            rle_ann (_type_): _description_
            image_size (_type_): (h, w, c)
            color (int, optional): _description_. Defaults to 1.
            binary (bool, optional): _description_. Defaults to True.
        """                
        
        if type(rle_ann['segmentation']['counts']) == list:
            rle = COCOMask.frPyObjects([rle_ann['segmentation']], h=image_size[0], w=image_size[1])
        else:
            rle = [rle_ann['segmentation']]
            
        mask = COCOMask.decode(rle) # (w, h, 1)
        
        # dimension order of pycocotools.mask rle decoding is different to cv2
        # Note there is a bug in the showAnns() of pycocotools which transpose the decoded rle mask and thus show inconsistent mask to the image
        mask = np.transpose(mask, (1, 0, 2)) # (h, w, 1)
        
        mask[mask>0] = color
        
        return mask    

color = 1
binary = True

parser = ArgumentParser(description="Semi-supervised segmentation")
parser.add_argument("--data_root", type=str, default="../Data/livecell-dataset/",
                        help="Root dir of datasets") 
args = parser.parse_args()

train_set_path = args.data_root + "livecell-dataset/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_train.json"
val_set_path = args.data_root + "livecell-dataset/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_val.json"
test_set_path = args.data_root + "livecell-dataset/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_test.json"

train_mask_path = args.data_root + "livecell-dataset/LIVECell_dataset_2021/masks/train-val/"
val_mask_path = args.data_root + "livecell-dataset/LIVECell_dataset_2021/masks/train-val/"
test_mask_path = args.data_root + "livecell-dataset/LIVECell_dataset_2021/masks/test/"

coco_parser = ParseCOCOAnn()
coco_parser.parse_dataset_annotation(train_set_path, train_mask_path, color, binary)
coco_parser.parse_dataset_annotation(val_set_path, val_mask_path, color, binary)
coco_parser.parse_dataset_annotation(test_set_path, test_mask_path, color, binary)
