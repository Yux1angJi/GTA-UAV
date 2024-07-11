export CUDA_VISIBLE_DEVICES=0,1
conda activate torch2

python train_visloc.py ./nohup_train_visloc_1234z3_group2_lpwb_ws_bs40_e5.out --gpu_ids 0,1 --label_smoothing 0.0 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_block" "part_block", "whole_slice"
python train_visloc.py ./nohup_train_visloc_1234z3_group2_lpwb_ps_bs40_e5.out --gpu_ids 0,1 --label_smoothing 0.0 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_block" "part_block", "part_slice"
python train_visloc.py ./nohup_train_visloc_1234z3_group2_lpb_ws_bs40_e5.out --gpu_ids 0,1 --label_smoothing 0.0 --batch_size 40 --train_in_group --group_len 2 --loss_type "part_block" "whole_slice"
python train_visloc.py ./nohup_train_visloc_1234z3_group2_lpb_ps_bs40_e5.out --gpu_ids 0,1 --label_smoothing 0.0 --batch_size 40 --train_in_group --group_len 2 --loss_type "part_block" "part_slice"
python train_visloc.py ./nohup_train_visloc_1234z3_group2_lwb_ps_bs40_e5.out --gpu_ids 0,1 --label_smoothing 0.0 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_block" "part_slice"
python train_visloc.py ./nohup_train_visloc_1234z3_group2_lwb_ws_bs40_e5.out --gpu_ids 0,1 --label_smoothing 0.0 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_block" "whole_slice"

python train_visloc.py ./nohup_train_visloc_1234z3_group2_lpwb_ws_s0.1_bs40_e5.out --gpu_ids 0,1 --label_smoothing 0.1 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_block" "part_block", "whole_slice"
python train_visloc.py ./nohup_train_visloc_1234z3_group2_lpwb_ps_s0.1_bs40_e5.out --gpu_ids 0,1 --label_smoothing 0.1 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_block" "part_block", "part_slice"
python train_visloc.py ./nohup_train_visloc_1234z3_group2_lpb_ws_s0.1_bs40_e5.out --gpu_ids 0,1 --label_smoothing 0.1 --batch_size 40 --train_in_group --group_len 2 --loss_type "part_block" "whole_slice"
python train_visloc.py ./nohup_train_visloc_1234z3_group2_lpb_ps_s0.1_bs40_e5.out --gpu_ids 0,1 --label_smoothing 0.1 --batch_size 40 --train_in_group --group_len 2 --loss_type "part_block" "part_slice"
python train_visloc.py ./nohup_train_visloc_1234z3_group2_lwb_ps_s0.1_bs40_e5.out --gpu_ids 0,1 --label_smoothing 0.1 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_block" "part_slice"
python train_visloc.py ./nohup_train_visloc_1234z3_group2_lwb_ws_s0.1_bs40_e5.out --gpu_ids 0,1 --label_smoothing 0.1 --batch_size 40 --train_in_group --group_len 2 --loss_type "whole_block" "whole_slice"