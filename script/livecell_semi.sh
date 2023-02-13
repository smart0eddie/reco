cd ../

python train_semisup.py --dataset livecell --num_labels 2 --label_ratio 2 --apply_reco --apply_aug classmix --data_root ../../../../../../Dataset/Medical/Segmentation/livecell-dataset/LIVECell_dataset_2021
python train_semisup.py --dataset livecell --num_labels 4 --label_ratio 4 --apply_reco --apply_aug classmix --data_root ../../../../../../Dataset/Medical/Segmentation/livecell-dataset/LIVECell_dataset_2021
python train_semisup.py --dataset livecell --num_labels 20 --label_ratio 20 --apply_reco --apply_aug classmix --data_root ../../../../../../Dataset/Medical/Segmentation/livecell-dataset/LIVECell_dataset_2021
python train_semisup.py --dataset livecell --num_labels 25 --label_ratio 25 --apply_reco --apply_aug classmix --data_root ../../../../../../Dataset/Medical/Segmentation/livecell-dataset/LIVECell_dataset_2021
python train_semisup.py --dataset livecell --num_labels 50 --label_ratio 50 --apply_reco --apply_aug classmix --data_root ../../../../../../Dataset/Medical/Segmentation/livecell-dataset/LIVECell_dataset_2021