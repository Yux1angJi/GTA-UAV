export CUDA_VISIBLE_DEVICES=0,1
# conda activate torch2

python train_visloc.py --log_to_file --log_path "nohup_train_visloc_1234z31_group2_lc_bs40_e5.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40


python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice"

python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_block"
python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_slice"
python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "whole_block"
python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "whole_slice"
python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_block" "whole_block"
python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_slice" "whole_slice"

python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_block" "whole_slice"
python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_slice" "whole_block"

# python train_visloc.py --gpu_ids 0,1 --label_smoothing 0.0 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "part_block"
# python train_visloc.py --gpu_ids 0,1 --label_smoothing 0.0 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_block"
# python train_visloc.py --gpu_ids 0,1 --label_smoothing 0.0 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "part_slice"
# python train_visloc.py --gpu_ids 0,1 --label_smoothing 0.0 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_block"

# python train_visloc.py --gpu_ids 0,1 --label_smoothing 0.0 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "part_block" "whole_block"
# python train_visloc.py --gpu_ids 0,1 --label_smoothing 0.0 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "part_slice" "whole_slice"
# python train_visloc.py --gpu_ids 0,1 --label_smoothing 0.0 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "part_block" "whole_slice"
# python train_visloc.py --gpu_ids 0,1 --label_smoothing 0.0 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "part_block" "part_slice"
# python train_visloc.py --gpu_ids 0,1 --label_smoothing 0.0 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_block" "part_slice"
# python train_visloc.py --gpu_ids 0,1 --label_smoothing 0.0 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_block" "whole_slice"

# python train_visloc.py --gpu_ids 0,1 --label_smoothing 0.0 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_block" "part_block" "whole_slice"
# python train_visloc.py --gpu_ids 0,1 --label_smoothing 0.0 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_block" "part_block" "part_slice"
# python train_visloc.py --gpu_ids 0,1 --label_smoothing 0.0 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_slice" "part_slice" "whole_block"
# python train_visloc.py --gpu_ids 0,1 --label_smoothing 0.0 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_slice" "part_slice" "part_block"

# python train_visloc.py --gpu_ids 0,1 --label_smoothing 0.0 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_slice" "part_slice" "whole_block" "part_block"