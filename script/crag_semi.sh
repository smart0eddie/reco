cd ../

python train_semisup.py --dataset crag --num_labels 2 --label_ratio 2 --apply_reco --apply_aug classmix --data_root ../../../../../../Dataset/Medical/Segmentation/CRAG/CRAG_v2/CRAG
python train_semisup.py --dataset crag --num_labels 4 --label_ratio 4 --apply_reco --apply_aug classmix --data_root ../../../../../../Dataset/Medical/Segmentation/CRAG/CRAG_v2/CRAG
python train_semisup.py --dataset crag --num_labels 8 --label_ratio 8 --apply_reco --apply_aug classmix --data_root ../../../../../../Dataset/Medical/Segmentation/CRAG/CRAG_v2/CRAG
python train_semisup.py --dataset crag --num_labels 16 --label_ratio 16 --apply_reco --apply_aug classmix --data_root ../../../../../../Dataset/Medical/Segmentation/CRAG/CRAG_v2/CRAG