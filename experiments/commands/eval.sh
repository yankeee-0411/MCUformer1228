cd /home/ubuntu/louis_crq/AutoFormer_MCU_louis0612
export CUDA_VISIBLE_DEVICES=0,1,2,3

output_dir=/home/ubuntu/louis_crq/AutoFormer_MCU_louis0612/output_louis_256KB/1210_300_eval_227486

mkdir -p $output_dir
output_file=$output_dir/output.txt

nohup python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port=29501 evolution.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012 --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T3.yaml \
--resume /home/ubuntu/louis_crq/AutoFormer_MCU_louis0612/output_louis_256KB/1210_300/iteration_0/model2_P22_R_attn74R_mlp86_updated.pth \
--output $output_dir --memory_limit 262144 --data-set EVO_IMNET > $output_file &

# Put this to cfg
# EVOLUTION:  
#   RANK_RATIO_ATTN: 0.84
#   RANK_RATIO_MLP: 0.86
#   PATCH_SIZE: 20