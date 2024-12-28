cd /home/ubuntu/louis_crq/AutoFormer_MCU_louis0612

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


nohup python -u -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012 --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 30 --warmup-epochs 5 \
--output ./output_louis0815 --batch-size 384 \
--pre_trained_model ./structure/supernet-tiny.pth --lr 1e-4 --start_iteration 0 &

nohup python -u -m torch.distributed.launch --nproc_per_node=4 --use_env supernet_train.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012 --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 300 --warmup-epochs 20 \
--output ./output_louis0819 --batch-size 384 \
--pre_trained_model ./structure/supernet-tiny.pth --lr 1e-4 --start_iteration 0 &

tmux

nohup python -u -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012 --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 30 --warmup-epochs 5 \
--output ./output_louis0826 --batch-size 384 \
--pre_trained_model ./structure/supernet-tiny.pth --lr 1e-4 --start_iteration 0 --limit_memory 256000 &

# wait

# nohup python -u -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012 --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 30 --warmup-epochs 5 \
# --output ./output_louis0813 --batch-size 384 \
# --pre_trained_model ./structure/supernet-tiny.pth --lr 1e-4 --start_iteration 0 &

# wait

# nohup python -u -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012 --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 30 --warmup-epochs 5 \
# --output ./output_louis0813 --batch-size 384 \
# --pre_trained_model ./structure/supernet-tiny.pth --lr 1e-4 --start_iteration 0 &

# wait

# nohup python -u -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012 --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 30 --warmup-epochs 5 \
# --output ./output_louis0813 --batch-size 384 \
# --pre_trained_model ./structure/supernet-tiny.pth --lr 1e-4 --start_iteration 0 &

# wait

# nohup python -u -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012 --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 30 --warmup-epochs 5 \
# --output ./output_louis0813 --batch-size 384 \
# --pre_trained_model ./structure/supernet-tiny.pth --lr 1e-4 --start_iteration 0 &

# wait

# nohup python -u -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012 --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 30 --warmup-epochs 5 \
# --output ./output_louis0813 --batch-size 384 \
# --pre_trained_model ./structure/supernet-tiny.pth --lr 1e-4 --start_iteration 0 &