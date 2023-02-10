from argparse import ArgumentParser
import string
import xml.etree.ElementTree as ET

import cv2
import numpy as np

def get_coordinate(vertices: ET.Element):
    pts = []
    
    for p in vertices:
        x = round(float(p.get('X')))
        y = round(float(p.get('Y')))
        
        pts.append((x, y))
        
    return pts

def parse_annotation(path: string):
    tree = ET.parse(path)
    root = tree.getroot()
    
    parsed_anns = []
    
    regions = root[0][1]
    
    for i, region in enumerate(regions):
        if 0 == i:
            continue
        
        zoom = region.get('Zoom')
        vertices = region[1]
        
        pts = get_coordinate(vertices)
        
        ann = dict(zoom=zoom, pts=pts)
        parsed_anns.append(ann)
        
    return parsed_anns       
        
def render_mask(anns, img_size, color=1, binary=True):
    """_summary_

    Args:
        anns (_type_): _description_
        img_size (_type_): (w, h)
    """

    mask = np.zeros(img_size, np.uint8)
    
    for idx, ann in enumerate(anns):
        c = color if binary else idx
        
        pts = [np.transpose(np.array([ann["pts"]]), (1, 0, 2))]
        mask = cv2.drawContours(mask, pts, contourIdx=-1, color=c, thickness=cv2.FILLED)

    return mask

img_size = (1000, 1000, 1)
color = 1

parser = ArgumentParser(description="Semi-supervised segmentation")
parser.add_argument("--data_root", type=str, default="../Data/MoNuSeg2018/",
                        help="Root dir of datasets") 
args = parser.parse_args()

train_set_path = args.data_root + "MoNuSeg2018/MoNuSeg 2018 Training Data/MoNuSeg 2018 Training Data/Annotations/"
test_set_path = args.data_root + "MoNuSeg2018/MoNuSegTestData/MoNuSegTestData/"

train_list = [
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

test_list = [    
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


for file in train_list:
    ann_path = train_set_path + file + ".xml"

    parsed_anns = parse_annotation(ann_path)
    print(f"file: {file}, anns: {len(parsed_anns)}")
    
    mask = render_mask(parsed_anns, img_size, color, binary=True)
    
    mask_path = train_set_path + file + "_b.png"
    cv2.imwrite(mask_path, mask)

for file in test_list:
    ann_path = test_set_path + file + ".xml"

    parsed_anns = parse_annotation(ann_path)
    print(f"file: {file}, anns: {len(parsed_anns)}")
    
    mask = render_mask(parsed_anns, img_size, color, binary=True)
    
    mask_path = test_set_path + file + "_b.png"
    cv2.imwrite(mask_path, mask)