cd /home/ubuntu/louis_crq/AutoFormer_MCU_louis0612

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup python -u -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012 --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-S.yaml --epochs 300 --warmup-epochs 10 --stop_epoch 100 \
--output ./output_louis0917 --batch-size 512 --input_size 240 \
--pre_trained_model ./structure/supernet-tiny.pth --lr 1e-4 --start_iteration 0 --limit_memory 256000 --seed 123 &
