cd /home/ubuntu/louis_crq/AutoFormer_MCU_louis0612

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7


# 0917 16 100 100 168
# 0918 normal run iteration
# 0919 24 100 100 168
# 0920 24 100 100 192
# 0921 16 100 100 192

# 262144
# 327680

# nohup python -u -m torch.distributed.launch --nproc_per_node=4 --use_env supernet_train.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012 --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T2.yaml --epochs 300 --warmup-epochs 5 --stop_epoch 30 \
# --output ./output_louis_256KB/1223 --batch-size 512 --input_size 224 \
# --pre_trained_model ./structure/supernet-tiny168.pth --lr 1e-4 --start_iteration 0 --limit_memory 262144 --seed 123 &

# /home/ubuntu/louis_crq/AutoFormer_MCU_louis0612/structure/supernet-tiny168.pth

output_path=./output_louis_256KB/1223
mkdir -p $output_path

nohup python -u -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 23220 supernet_train.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012 --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T2.yaml --epochs 300 --warmup-epochs 5 --stop_epoch 30 \
--output $output_path --batch-size 512 --input_size 224 \
--pre_trained_model ./structure/supernet-tiny.pth --lr 1e-4 --start_iteration 1 --limit_memory 262144 --seed 123 >> $output_path/output.log 2>&1 &
