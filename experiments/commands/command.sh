# Environments
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116


# Locations
/home/wzw/louis_cyk/For_mount/ILSVRC2012
/home/wzw/louis_cyk/For_mount/ILSVRC2012
sudo mount /dev/sdf /home/wzw/louis_cyk/For_mount #227
sudo mount --bind /data1 /home/ubuntu/louis_crq/For_mount #27


# Training Commands
python createnewsupernet.py --change_qk --relative_position --mode super --cfg ./experiments/supernet/supernet-T.yaml 

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
export OMP_NUM_THREADS=2
cd /home/ubuntu/louis_crq/AutoFormer_MCU_louis0612 \
conda activate louis_MCUformer 

nohup python -u -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012 --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 30 --warmup-epochs 5 \
--output ./output_louis1 --batch-size 320 \
--pre_trained_model ./structure/supernet-tiny.pth --lr 2e-4 --start_iteration 0 &

nohup python supernet_train.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012 --gp \
--change_qk --relative_position --mode super --cfg ./experiments/supernet/supernet-T.yaml --epochs 30 --warmup-epochs 5 \
--output ./output_louis0 --batch-size 320 \
--pre_trained_model ./structure/supernet-tiny.pth --lr 2e-4 --start_iteration 0 &

# Search Commands
python ./lib/subImageNet.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012

nohup python -m torch.distributed.launch --nproc_per_node=4 --use_env evolution.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012 --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml \
--resume /home/ubuntu/louis_crq/AutoFormer_MCU_louis0612/output_louis3/iteration_0/model_P20_R0_updated.pth \
--checkpoint_path /home/ubuntu/louis_crq/AutoFormer_MCU_louis0612/evo_output_louis4/checkpoint-5.pth.tar \
--output ./evo_output_louis4 --memory_limit 320000 --data-set EVO_IMNET &

nohup python -m torch.distributed.launch --nproc_per_node=4 --use_env evolution.py --data-path /home/ubuntu/louis_crq/For_mount/ILSVRC2012 --gp \
--change_qk --relative_position --dist-eval --cfg /home/ubuntu/louis_crq/AutoFormer_MCU_louis0612/output_louis0820/cfg.yaml \
--resume /home/ubuntu/louis_crq/AutoFormer_MCU_louis0612/output_louis0820/iteration_0/model0_P16_R_attn70R_mlp85_updated.pth \
--checkpoint_path /home/ubuntu/louis_crq/AutoFormer_MCU_louis0612/evo_output_louis0819/checkpoint-6.pth.tar \
--output ./evo_output_louis0819 --memory_limit 320000 --data-set EVO_IMNET &