#!/bin/sh

# Cross-area setting with weighted-InfoNCE k=5
python train_gta.py --data_root <The directory of the GTA-UAV dataset> --train_pairs_meta_file "cross-area-drone2sate-train.json" --test_pairs_meta_file "cross-area-drone2sate-test.json" --gpu_ids 0,1 --label_smoothing 0.05 --with_weight --k 5 --epoch 5 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64

# Cross-area setting with InfoNCE
python train_gta.py --data_root <The directory of the GTA-UAV dataset> --train_pairs_meta_file "cross-area-drone2sate-train.json" --test_pairs_meta_file "cross-area-drone2sate-test.json" --gpu_ids 0,1 --label_smoothing 0.05 --epoch 5 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64


# Same-area setting with weighted-InfoNCE k=5
python train_gta.py --data_root <The directory of the GTA-UAV dataset> --train_pairs_meta_file "same-area-drone2sate-train.json" --test_pairs_meta_file "same-area-drone2sate-test.json" --gpu_ids 0,1 --label_smoothing 0.05 --with_weight --k 5 --epoch 20 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64

# Same-area setting with InfoNCE
python train_gta.py --data_root <The directory of the GTA-UAV dataset> --train_pairs_meta_file "same-area-drone2sate-train.json" --test_pairs_meta_file "same-area-drone2sate-test.json" --gpu_ids 0,1 --label_smoothing 0.05 --epoch 20 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64
