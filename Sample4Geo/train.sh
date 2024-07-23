export CUDA_VISIBLE_DEVICES=3
# conda activate torch2

# python train_gta.py --data_dir "randcam2_std0_stable_all/cross_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_z41_cross_allsemiiou4_lc_bs40_e5_s0.1k3_valiou_vit.out" --train_mode 'semi_iou' --test_mode 'iou' --gpu_ids 0 --label_smoothing 0.1 --k 3 --epoch 5 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/cross_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_z41_cross_allsemiiou4_lc_bs64_e10_s0.1k3_valiou_vit.out" --train_mode 'semi_iou' --test_mode 'iou' --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 3 --epoch 10 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/cross_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_z41_cross_allsemiiou4_group2_lcspbwb_bs64_e10_s0.1_valiou_vit.out" --train_mode 'semi_iou' --test_mode 'iou' --gpu_ids 0 --label_smoothing 0.1 --epoch 10 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_block" "whole_block"
# python train_gta.py --data_dir "randcam2_std0_stable_all/cross_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_z41_cross_allsemiiou4_lc_bs64_e10_s0.1_valiou_vit.out" --train_mode 'semi_iou' --test_mode 'iou' --gpu_ids 0 --label_smoothing 0.1 --epoch 10 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/corss_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_z41_cross_allsemiiou4_lc_bs64_e10_valiou_vit.out" --train_mode 'semi_iou' --test_mode 'iou' --gpu_ids 0 --epoch 10 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64



# python train_gta.py --data_dir "randcam2_std0_stable_all/same_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_z41_same_allsemiiou4_lc_bs64_e10_s0.1k3_valiou_vit.out" --train_mode 'semi_iou' --test_mode 'iou' --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 3 --epoch 10 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/same_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_z41_same_allsemiiou4_group2_lcspbwb_bs64_e10_s0.1_valiou_vit.out" --train_mode 'semi_iou' --test_mode 'iou' --gpu_ids 0 --label_smoothing 0.1 --epoch 10 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_block" "whole_block"
# python train_gta.py --data_dir "randcam2_std0_stable_all/same_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_z41_same_allsemiiou4_lc_bs64_e10_s0.1_valiou_vit.out" --train_mode 'semi_iou' --test_mode 'iou' --gpu_ids 0 --label_smoothing 0.1 --epoch 10 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/same_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_z41_same_allsemiiou4_lc_bs64_e10_valiou_vit.out" --train_mode 'semi_iou' --test_mode 'iou' --gpu_ids 0 --epoch 10 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001 --batch_size 64


python train_visloc.py --data_dir "same_all_iou4_oc4_z31" --log_to_file --log_path "nohup_train_visloc_allsemiiou_ls_bs64_e5_s0.1k3_valiou_vit_fromgta_freeze33270.out" --train_mode 'semi_iou' --test_mode 'iou' --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 0.3 --epoch 5 --batch_size 64 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --checkpoint_start "work_dir/gta/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0722110449/weights_end.pth" --lr 0.0001 --freeze_layers --frozen_stages 3 3 27 0
python train_visloc.py --data_dir "same_all_iou4_oc4_z31" --log_to_file --log_path "nohup_train_visloc_allsemiiou_ls_bs64_e5_s0.1k3_valiou_vit.out" --train_mode 'semi_iou' --test_mode 'iou' --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 0.3 --epoch 5 --batch_size 64 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001
python train_visloc.py --data_dir "same_all_iou4_oc4_z31" --log_to_file --log_path "nohup_train_visloc_allsemiiou_ls_bs64_e5_s0.1k3_valiou_vit_fromuniversity_freeze33270.out" --train_mode 'semi_iou' --test_mode 'iou' --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 0.3 --epoch 5 --batch_size 64 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --checkpoint_start "work_dir/university/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/weights_end.pth" --lr 0.0001 --freeze_layers --frozen_stages 3 3 27 0
python train_visloc.py --data_dir "same_all_iou4_oc4_z31" --log_to_file --log_path "nohup_train_visloc_allsemiiou_ls_bs64_e5_s0.1k3_valiou_vit_fromdenseuav_freeze33270.out" --train_mode 'semi_iou' --test_mode 'iou' --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 0.3 --epoch 5 --batch_size 64 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --checkpoint_start "work_dir/denseuav/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0723164458/weights_end.pth" --lr 0.0001 --freeze_layers --frozen_stages 3 3 27 0
python train_visloc.py --data_dir "same_all_iou4_oc4_z31" --log_to_file --log_path "nohup_train_visloc_allsemiiou_ls_bs64_e5_s0.1k3_valiou_vit_fromsues200_freeze33270.out" --train_mode 'semi_iou' --test_mode 'iou' --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 0.3 --epoch 5 --batch_size 64 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --checkpoint_start "work_dir/sues/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0723160833/weights_end.pth" --lr 0.0001 --freeze_layers --frozen_stages 3 3 27 0





# python train_visloc.py --data_dir "same_all_iou4_oc4_z31" --log_to_file --log_path "nohup_train_visloc_allsemiiou_ls_bs64_e5_s0.1k3_valiou_vit_fromgta_freeze3300.out" --train_mode 'semi_iou' --test_mode 'iou' --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 0.3 --epoch 5 --batch_size 64 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --checkpoint_start "work_dir/gta/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0721180833/weights_e3_0.6296.pth" --lr 0.0001 --freeze_layers --frozen_stages 3 3 0 0

# python train_visloc.py --data_dir "same_all_iou4_oc4_z31" --log_to_file --log_path "nohup_train_visloc_allsemiiou_ls_bs64_e5_s0.1k3_valiou_vit.out" --train_mode 'semi_iou' --test_mode 'iou' --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 0.3 --epoch 5 --batch_size 64 --model 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k' --lr 0.0001


# python train_gta.py --data_dir "randcam2_std0_stable/test" --log_to_file --log_path "nohup_test.out" --gpu_ids 0 --label_smoothing 0.1 --k 3 --epoch 10 --batch_size 64
# python train_visloc.py --data_dir "data_all_iou4_oc4_z3" --log_to_file --log_path "nohup_train_visloc_alloc4_group2_lcspbws_bs64_e5_s0.1_valoc.out" --train_mode 'oc' --test_mode 'oc' --gpu_ids 0 --label_smoothing 0.1 --epoch 5 --batch_size 64 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_block" "whole_slice"
# python train_visloc.py --data_dir "data_all_iou4_oc4_z3" --log_to_file --log_path "nohup_train_visloc_alloc4_group2_lcspbwb_bs64_e5_s0.1_valoc.out" --train_mode 'oc' --test_mode 'oc' --gpu_ids 0 --label_smoothing 0.1 --epoch 5 --batch_size 64 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_block" "whole_block"
# python train_visloc.py --data_dir "data_all_iou4_oc4_z3" --log_to_file --log_path "nohup_train_visloc_alliou4_lc_bs64_e5_s0.1k3_valiou.out" --train_mode 'iou' --test_mode 'iou' --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 3 --epoch 5 --batch_size 64
# python train_visloc.py --data_dir "data_all_iou4_oc4_z3" --log_to_file --log_path "nohup_train_visloc_alloc4_lc_bs64_e5_s0.1k3_valoc.out" --train_mode 'oc' --test_mode 'oc' --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 3 --epoch 5 --batch_size 64

# python train_visloc.py --data_dir "same_all_iou4_oc4_z31" --log_to_file --log_path "nohup_train_visloc_sameoc4_lc_bs64_e5_s0.1k3_valoc.out" --train_mode 'oc' --test_mode 'oc' --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 3 --epoch 5 --batch_size 64
# python train_visloc.py --data_dir "same_all_iou4_oc4_z31" --log_to_file --log_path "nohup_train_visloc_sameoc4_group2_lcpbws_bs64_e5_s0.1_valoc.out" --train_mode 'oc' --test_mode 'oc' --gpu_ids 0 --label_smoothing 0.1 --epoch 5 --batch_size 64 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_block" "whole_slice"

# python train_visloc.py --data_dir "data_all_iou4_oc4_z3" --log_to_file --log_path "nohup_train_visloc_alloc4_lc_bs64_e5_s0.1k3_valoc_fromgta_freeze33270.out" --checkpoint_start "work_dir/gta/convnext_base.fb_in22k_ft_in1k_384/0719000212/weights_end.pth" --freeze_layers --frozen_stages 3 3 27 0 --train_mode 'oc' --test_mode 'oc' --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 3 --epoch 5 --batch_size 64
# python train_visloc.py --data_dir "data_all_iou4_oc4_z3" --log_to_file --log_path "nohup_train_visloc_alloc4_lc_bs64_e5_s0.1k3_valoc_fromgta_freeze3300.out" --checkpoint_start "work_dir/gta/convnext_base.fb_in22k_ft_in1k_384/0719000212/weights_end.pth" --freeze_layers --frozen_stages 3 3 0 0 --train_mode 'oc' --test_mode 'oc' --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 3 --epoch 5 --batch_size 64

# python train_gta.py --data_dir "randcam2_std0_stable_all/cross_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_crossiou4_lc_bs64_e20_s0.1_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --label_smoothing 0.1 --epoch 20 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/cross_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_crossiou4_lc_bs64_e20_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --epoch 20 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/cross_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_crossiou4_lc_bs64_e20_k0.5_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --with_weight --k 0.5 --epoch 20 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/cross_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_crossiou4_lc_bs64_e20_s0.1k0.5_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 0.5 --epoch 20 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/cross_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_crossiou4_lc_bs64_e20_s0.1k1_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 1 --epoch 20 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/cross_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_crossiou4_lc_bs64_e20_s0.1k3_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 3 --epoch 20 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/cross_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_crossiou4_lc_bs64_e20_s0.1k5_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 5 --epoch 20 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/cross_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_crossiou4_lc_bs64_e20_s0.1k7_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 7 --epoch 20 --batch_size 64


# python train_gta.py --data_dir "randcam2_std0_stable_all/same_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_sameiou4_lc_bs64_e20_s0.1_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --label_smoothing 0.1 --epoch 20 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/same_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_sameiou4_lc_bs64_e20_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --epoch 20 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/same_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_sameiou4_lc_bs64_e20_k0.5_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --with_weight --k 0.5 --epoch 20 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/same_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_sameiou4_lc_bs64_e20_s0.1k0.5_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 0.5 --epoch 20 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/same_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_sameiou4_lc_bs64_e20_s0.1k1_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 1 --epoch 20 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/same_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_sameiou4_lc_bs64_e20_s0.1k3_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 3 --epoch 20 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/same_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_sameiou4_lc_bs64_e20_s0.1k5_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 5 --epoch 20 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/same_h23456_z41_iou4_oc4" --log_to_file --log_path "nohup_train_gta_sameiou4_lc_bs64_e20_s0.1k7_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 7 --epoch 20 --batch_size 64


# python train_gta.py --data_dir "randcam2_std0_stable_all/train_h23456_z4_iou4_oc4" --log_to_file --log_path "nohup_train_gta_alliou4_lc_bs64_e10_s0.1k3_valoc.out" --train_mode "iou" --test_mode "oc" --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 3 --epoch 10 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/train_h23456_z4_iou4_oc4" --log_to_file --log_path "nohup_train_gta_alliou4_lc_bs64_e10_s0.1_valoc.out" --train_mode "iou" --test_mode "oc" --gpu_ids 0 --label_smoothing 0.1 --epoch 10 --batch_size 64


# python train_gta.py --data_dir "randcam2_std0_stable_all/train_h23456_z4_iou4_oc4" --log_to_file --log_path "nohup_train_gta_alloc4_group2_lcspbws_bs64_e5_s0.1_valoc.out" --train_mode "oc" --test_mode "oc" --gpu_ids 0 --label_smoothing 0.1 --epoch 5 --batch_size 64 --train_in_group --group_len 2 --loss_type "contrastive_slice" "part_block" "whole_slice"
# python train_gta.py --data_dir "randcam2_std0_stable_all/train_h23456_z4_iou4_oc4" --log_to_file --log_path "nohup_train_gta_alliou4_lc_bs64_e5_s0.1k3_valoc.out" --train_mode "iou" --test_mode "oc" --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 3 --epoch 5 --batch_size 64
# python train_gta.py --data_dir "randcam2_std0_stable_all/train_h23456_z4_iou4_oc4" --log_to_file --log_path "nohup_train_gta_alliou4_lc_bs64_e5_s0.1k3_valiou.out" --train_mode "iou" --test_mode "iou" --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 3 --epoch 5 --batch_size 64

# python train_visloc.py --data_dir "data_all_iou4" --log_to_file --log_path "nohup_train_visloc_alliou4_lc_bs64_e5_s0.1k3.out"  --gpu_ids 0 --label_smoothing 0.1 --k 3 --epoch 5 --batch_size 64
# python train_visloc.py --data_dir "data_all_iou4" --log_to_file --log_path "nohup_train_visloc_alliou4_lc_bs64_e5_s0.1k5.out"  --gpu_ids 0 --label_smoothing 0.1 --k 5 --epoch 5 --batch_size 64
# python train_visloc.py --data_dir "data_all_iou4" --log_to_file --log_path "nohup_train_visloc_alliou4_lc_bs64_e5_s0.1k10.out"  --gpu_ids 0 --label_smoothing 0.1 --k 10 --epoch 5 --batch_size 64


# python train_visloc.py --data_dir "data_all4_iou4" --log_to_file --log_path "nohup_train_visloc_all4iou4_lc_bs64_e10_s0.1k3_fromgta_freeze33270.out" --checkpoint_start "work_dir/gta/convnext_base.fb_in22k_ft_in1k_384/0715230053/weights_end.pth" --freeze_layers --frozen_stages 3 3 27 0 --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 3 --epoch 10 --batch_size 64
# python train_visloc.py --data_dir "data_all4_iou4" --log_to_file --log_path "nohup_train_visloc_all4iou4_lc_bs64_e10_s0.1k3_fromuni_freeze33270.out" --checkpoint_start "pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth" --freeze_layers --frozen_stages 3 3 27 0 --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 3 --epoch 10 --batch_size 64
# python train_visloc.py --data_dir "data_all4_iou4" --log_to_file --log_path "nohup_train_visloc_all4iou4_lc_bs64_e10_s0.1k3.out" --gpu_ids 0 --label_smoothing 0.1 --with_weight --k 3 --epoch 10 --batch_size 64

# python train_visloc.py --data_dir "data_all4_iou4" --log_to_file --log_path "nohup_train_visloc_all4iou4_lc_bs64_e10_s0.1_fromgta_freeze33270.out" --checkpoint_start "work_dir/gta/convnext_base.fb_in22k_ft_in1k_384/0715230053/weights_end.pth" --freeze_layers --frozen_stages 3 3 27 0 --gpu_ids 0 --label_smoothing 0.1 --epoch 10 --batch_size 64
# python train_visloc.py --data_dir "data_all4_iou4" --log_to_file --log_path "nohup_train_visloc_all4iou4_lc_bs64_e10_s0.1_fromuni_freeze33270.out" --checkpoint_start "pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth" --freeze_layers --frozen_stages 3 3 27 0 --gpu_ids 0 --label_smoothing 0.1 --epoch 10 --batch_size 64
# python train_visloc.py --data_dir "data_all4_iou4" --log_to_file --log_path "nohup_train_visloc_all4iou4_lc_bs64_e10_s0.1.out" --gpu_ids 0 --label_smoothing 0.1 --epoch 10 --batch_size 64

# python train_visloc.py --data_dir "data_all4_iou4" --log_to_file --log_path "nohup_train_visloc_all4iou4_lc_bs40_e5.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40
# python train_visloc.py --data_dir "data_all4_iou4" --checkpoint_start "work_dir/gta/convnext_base.fb_in22k_ft_in1k_384/0714140708/weights_end.pth" --log_to_file --log_path "nohup_train_visloc_all4iou4_lc_bs40_e5_fromgta.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40
# python train_visloc.py --data_dir "data_all4_iou4" --checkpoint_start "pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth" --log_to_file --log_path "nohup_train_visloc_all4iou4_lc_bs40_e5_fromuni.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40

# python train_visloc.py --data_dir "data_all8_iou4" --log_to_file --log_path "nohup_train_visloc_all8iou4_lc_bs40_e5.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40
# python train_visloc.py --data_dir "data_all8_iou4" --checkpoint_start "work_dir/gta/convnext_base.fb_in22k_ft_in1k_384/0714140708/weights_end.pth" --log_to_file --log_path "nohup_train_visloc_all8iou4_lc_bs40_e5_fromgta.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40
# python train_visloc.py --data_dir "data_all8_iou4" --checkpoint_start "pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth" --log_to_file --log_path "nohup_train_visloc_all8iou4_lc_bs40_e5_fromuni.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40

# python train_visloc.py --data_dir "data_all4_iou3" --log_to_file --log_path "nohup_train_visloc_all4iou3_lc_bs40_e5.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40
# python train_visloc.py --data_dir "data_all4_iou3" --checkpoint_start "work_dir/gta/convnext_base.fb_in22k_ft_in1k_384/0715023821/weights_end.pth" --log_to_file --log_path "nohup_train_visloc_all4iou3_lc_bs40_e5_fromgta.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40
# python train_visloc.py --data_dir "data_all4_iou3" --checkpoint_start "pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth" --log_to_file --log_path "nohup_train_visloc_all4iou3_lc_bs40_e5_fromuni.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40

# python train_visloc.py --data_dir "data_all8_iou3" --log_to_file --log_path "nohup_train_visloc_all8iou3_lc_bs40_e5.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40
# python train_visloc.py --data_dir "data_all8_iou3" --checkpoint_start "work_dir/gta/convnext_base.fb_in22k_ft_in1k_384/0715023821/weights_end.pth" --log_to_file --log_path "nohup_train_visloc_all8iou3_lc_bs40_e5_fromgta.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40
# python train_visloc.py --data_dir "data_all8_iou3" --checkpoint_start "pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth" --log_to_file --log_path "nohup_train_visloc_all8iou3_lc_bs40_e5_fromuni.out" --gpu_ids 0,1 --label_smoothing 0.1 --epoch 5 --batch_size 40

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