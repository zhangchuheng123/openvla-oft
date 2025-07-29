# torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
torchrun --standalone --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /mnt/chuheng_data/robot_ft_data/data_v6/data_v6_combined/processed/ \
  --dataset_name data_v6_combined \
  --run_root_dir /mnt/chuheng_data/exp_pi/250727_openvlaoft \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --run_id_note none