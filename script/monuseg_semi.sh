cd ../

python train_semisup.py --dataset monuseg --num_labels 2 --label_ratio 2 --apply_reco --apply_aug classmix --data_root ../../../../../../Dataset/Medical/Segmentation/MoNuSeg2018
python train_semisup.py --dataset monuseg --num_labels 4 --label_ratio 4 --apply_reco --apply_aug classmix --data_root ../../../../../../Dataset/Medical/Segmentation/MoNuSeg2018
python train_semisup.py --dataset monuseg --num_labels 8 --label_ratio 8 --apply_reco --apply_aug classmix --data_root ../../../../../../Dataset/Medical/Segmentation/MoNuSeg2018
python train_semisup.py --dataset monuseg --num_labels 16 --label_ratio 16 --apply_reco --apply_aug classmix --data_root ../../../../../../Dataset/Medical/Segmentation/MoNuSeg2018