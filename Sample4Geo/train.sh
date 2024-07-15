export CUDA_VISIBLE_DEVICES=0,1
# conda activate torch2

python train_visloc.py --data_dir "data_all4_iou4" --log_to_file --log_path "nohup_train_visloc_all4iou4_lc_bs40_e5.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40
python train_visloc.py --data_dir "data_all4_iou4" --checkpoint_start "work_dir/gta/convnext_base.fb_in22k_ft_in1k_384/0714140708/weights_end.pth" --log_to_file --log_path "nohup_train_visloc_all4iou4_lc_bs40_e5_fromgta.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40
python train_visloc.py --data_dir "data_all4_iou4" --checkpoint_start "pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth" --log_to_file --log_path "nohup_train_visloc_all4iou4_lc_bs40_e5_fromuni.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40

python train_visloc.py --data_dir "data_all8_iou4" --log_to_file --log_path "nohup_train_visloc_all8iou4_lc_bs40_e5.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40
python train_visloc.py --data_dir "data_all8_iou4" --checkpoint_start "work_dir/gta/convnext_base.fb_in22k_ft_in1k_384/0714140708/weights_end.pth" --log_to_file --log_path "nohup_train_visloc_all8iou4_lc_bs40_e5_fromgta.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40
python train_visloc.py --data_dir "data_all8_iou4" --checkpoint_start "pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth" --log_to_file --log_path "nohup_train_visloc_all8iou4_lc_bs40_e5_fromuni.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40

python train_visloc.py --data_dir "data_all4_iou3" --log_to_file --log_path "nohup_train_visloc_all4iou3_lc_bs40_e5.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40
python train_visloc.py --data_dir "data_all4_iou3" --checkpoint_start "work_dir/gta/convnext_base.fb_in22k_ft_in1k_384/0715023821/weights_end.pth" --log_to_file --log_path "nohup_train_visloc_all4iou3_lc_bs40_e5_fromgta.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40
python train_visloc.py --data_dir "data_all4_iou3" --checkpoint_start "pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth" --log_to_file --log_path "nohup_train_visloc_all4iou3_lc_bs40_e5_fromuni.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40

python train_visloc.py --data_dir "data_all8_iou3" --log_to_file --log_path "nohup_train_visloc_all8iou3_lc_bs40_e5.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40
python train_visloc.py --data_dir "data_all8_iou3" --checkpoint_start "work_dir/gta/convnext_base.fb_in22k_ft_in1k_384/0715023821/weights_end.pth" --log_to_file --log_path "nohup_train_visloc_all8iou3_lc_bs40_e5_fromgta.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40
python train_visloc.py --data_dir "data_all8_iou3" --checkpoint_start "pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth" --log_to_file --log_path "nohup_train_visloc_all8iou3_lc_bs40_e5_fromuni.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40

# python train_visloc.py --log_to_file --log_path "nohup_train_visloc_1234z3_group2_lc_bs40_e5.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40

# python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice"

# python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_block"
# python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_slice"
# python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "whole_block"
# python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "whole_slice"
# python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_block" "whole_block"
# python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_slice" "whole_slice"

# python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_block" "whole_slice"
# python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_slice" "whole_block"
# python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_slice" "part_block"
# python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "whole_slice" "whole_block"

# python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_slice" "whole_slice" "part_block"
# python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_slice" "whole_slice" "whole_block"
# python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_block" "whole_block" "whole_slice"
# python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_block" "whole_block" "part_slice"

# python train_visloc.py --log_to_file --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_slice" "whole_slice" "part_block" "whole_block"

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