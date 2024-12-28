import argparse
import datetime
import itertools
import json
import os
import random
import time
from pathlib import Path
import math

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler

from lib import utils
from lib.config import cfg, update_config_from_file
from lib.datasets import build_dataset
from lib.samplers import RASampler
from model.supernet_transformer import Vision_TransformerSuper
from structure.update_structure import low_rank_decompo
from structure.write_structure import write_model_structure
from supernet_engine import evaluate, sample_configs, train_one_epoch

from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression


def get_args_parser():
    parser = argparse.ArgumentParser(
        "AutoFormer training and evaluation script", add_help=False
    )
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    # config file
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )

    # custom parameters
    parser.add_argument(
        "--platform",
        default="pai",
        type=str,
        choices=["itp", "pai", "aml"],
        help="Name of model to train",
    )
    parser.add_argument(
        "--teacher_model", default="", type=str, help="Name of teacher model to train"
    )
    parser.add_argument("--relative_position", action="store_true")
    parser.add_argument("--gp", action="store_true")
    parser.add_argument("--change_qkv", action="store_true")
    parser.add_argument(
        "--max_relative_position",
        type=int,
        default=14,
        help="max distance in relative position embedding",
    )

    # Model parameters
    parser.add_argument(
        "--model", default="", type=str, metavar="MODEL", help="Name of model to train"
    )
    # AutoFormer config
    parser.add_argument(
        "--mode",
        type=str,
        default="super",
        choices=["super", "retrain"],
        help="mode of AutoFormer",
    )
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--patch_size", default=16, type=int)

    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )
    parser.add_argument(
        "--drop-block",
        type=float,
        default=None,
        metavar="PCT",
        help="Drop block rate (default: None)",
    )

    parser.add_argument("--model-ema", action="store_true")
    parser.add_argument(
        "--no-model-ema", action="store_false", dest="model_ema")
    # parser.set_defaults(model_ema=True)
    parser.add_argument("--model-ema-decay", type=float,
                        default=0.99996, help="")
    parser.add_argument(
        "--model-ema-force-cpu", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--rpe_type", type=str, default="bias", choices=["bias", "direct"]
    )
    parser.add_argument("--post_norm", action="store_true")
    parser.add_argument("--no_abs_pos", action="store_true")

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    # Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )
    parser.add_argument(
        "--lr-power",
        type=float,
        default=1.0,
        help="power of the polynomial lr scheduler",
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    parser.add_argument("--repeated-aug", action="store_true")
    parser.add_argument("--no-repeated-aug",
                        action="store_false", dest="repeated_aug")

    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.8,
        help="mixup alpha, mixup enabled if > 0. (default: 0.8)",
    )
    parser.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)",
    )
    parser.add_argument(
        "--cutmix-minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup-switch-prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup-mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # Dataset parameters
    parser.add_argument(
        "--data-path", default="./data/imagenet/", type=str, help="dataset path"
    )
    parser.add_argument(
        "--data-set",
        default="IMNET",
        choices=["CIFAR", "IMNET", "INAT", "INAT19"],
        type=str,
        help="Image Net dataset path",
    )
    parser.add_argument(
        "--inat-category",
        default="name",
        choices=[
            "kingdom",
            "phylum",
            "class",
            "order",
            "supercategory",
            "family",
            "genus",
            "name",
        ],
        type=str,
        help="semantic granularity",
    )

    parser.add_argument(
        "--output_dir", default="./", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true",
                        help="Perform evaluation only")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--dist-eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation",
    )
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false",
                        dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.set_defaults(amp=True)

    parser.add_argument("--pre_trained_model", default="", type=str)
    parser.add_argument("--supernet_num", default=10, type=int)
    parser.add_argument("--iteration", default=10, type=int)

    parser.add_argument("--sample_num", default=10, type=int)
    parser.add_argument("--upper_bound_prob", default=0.01, type=float)
    parser.add_argument("--limit_memory", default=320e3, type=float)

    parser.add_argument("--start_iteration", default=-1, type=int)
    parser.add_argument("--stop_epoch", default=30, type=int)
    parser.add_argument("--special_num", default=100, type=int)

    return parser


def main(args):
    utils.init_distributed_mode(args)
    update_config_from_file(args.cfg)

    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=int(2 * args.batch_size),
        sampler=sampler_val,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    random.seed(args.seed)

    choices = {
        "num_heads": cfg.SEARCH_SPACE.NUM_HEADS,
        "mlp_ratio": cfg.SEARCH_SPACE.MLP_RATIO,
        "embed_dim": cfg.SEARCH_SPACE.EMBED_DIM,
        "depth": cfg.SEARCH_SPACE.DEPTH,
    }

    output_dir = args.output_dir
    if (
        os.path.exists(os.path.join(output_dir, "log.yaml"))
        and args.start_iteration >= 0
    ):
        with open(os.path.join(output_dir, "log.yaml"), "r") as f:
            data = yaml.safe_load(f)
            chosen_config = data[f"iteration_{args.start_iteration}"]
        print(f"Continue from iteration {args.start_iteration}")

    else:
        chosen_config = []
        while (len(chosen_config) < 10):
            patch_size = random.choice(cfg.SUPERNET.PATCH_SIZE)
            rank_ratio_attn = random.choice(cfg.SUPERNET.RANK_RATIO)
            rank_ratio_mlp = random.choice(cfg.SUPERNET.RANK_RATIO)
            if [patch_size, rank_ratio_attn, rank_ratio_mlp] not in chosen_config:
                if check_valid_subnet_num(args, patch_size, rank_ratio_attn, rank_ratio_mlp, cfg, choices, sample_max_iter=1e3):
                    chosen_config.append(
                        [patch_size, rank_ratio_attn, rank_ratio_mlp])

    # linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    # args.lr = linear_scaled_lr

    for iter in range(max(0, args.start_iteration), args.iteration, 1):
        print(f"\nStart Iteration {iter}")
        print(f"Chosen config:")
        for k in range(len(chosen_config)):
            print(f"{k}: {chosen_config[k]}")

        if (
            not os.path.exists(os.path.join(args.output_dir, "log.yaml"))
        ) or iter > args.start_iteration:
            if utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.yaml"), "a") as f:
                    yaml.dump({f"iteration_{iter}": chosen_config},
                              f, default_flow_style=False)

        for model_num in range(len(chosen_config)):
            print(f"\nCreating SuperVisionTransformer")
            print(f"chosen_config: {chosen_config[model_num]}")

            P = int(chosen_config[model_num][0])
            R_attn = int(chosen_config[model_num][1] * 100)
            R_mlp = int(chosen_config[model_num][2] * 100)

            model = Vision_TransformerSuper(
                img_size=args.input_size,
                patch_size=int(chosen_config[model_num][0]),
                embed_dim=cfg.SUPERNET.EMBED_DIM,
                depth=cfg.SUPERNET.DEPTH,
                num_heads=cfg.SUPERNET.NUM_HEADS,
                mlp_ratio=cfg.SUPERNET.MLP_RATIO,
                qkv_bias=True,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                gp=args.gp,
                num_classes=args.nb_classes,
                max_relative_position=args.max_relative_position,
                relative_position=args.relative_position,
                change_qkv=args.change_qkv,
                abs_pos=not args.no_abs_pos,
                rank_ratio_attn=chosen_config[model_num][1],
                rank_ratio_mlp=chosen_config[model_num][2],
            )

            model.to(device)

            if args.teacher_model:
                teacher_model = create_model(
                    args.teacher_model,
                    pretrained=True,
                    num_classes=args.nb_classes,
                )
                teacher_model.to(device)
                teacher_loss = LabelSmoothingCrossEntropy(
                    smoothing=args.smoothing)
            else:
                teacher_model = None
                teacher_loss = None

            model_ema = None

            model_without_ddp = model

            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu], find_unused_parameters=True
                )
                model_without_ddp = model.module

            n_parameters = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
            print("number of params:", n_parameters)

            # Not here as the it is in the loop
            # linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
            # args.lr = linear_scaled_lr

            optimizer = create_optimizer(args, model_without_ddp)
            loss_scaler = NativeScaler()
            lr_scheduler, _ = create_scheduler(args, optimizer)

            # criterion = LabelSmoothingCrossEntropy()

            if args.mixup > 0.0:
                # smoothing is handled with mixup label transform
                criterion = SoftTargetCrossEntropy()
            elif args.smoothing:
                criterion = LabelSmoothingCrossEntropy(
                    smoothing=args.smoothing)
            else:
                criterion = torch.nn.CrossEntropyLoss()

            output_dir = Path(args.output_dir)
            if not output_dir.exists() and utils.is_main_process():
                output_dir.mkdir(parents=True)
            output_dir = output_dir / f"iteration_{iter}"
            if not output_dir.exists() and utils.is_main_process():
                output_dir.mkdir(parents=True)
            while not output_dir.exists():
                time.sleep(1)

            ReDecompose_flag = False

            if os.path.exists(
                os.path.join(
                    output_dir,
                    f"model{model_num}_P{P}_R_attn{R_attn}R_mlp{R_mlp}_updated.pth",
                )
            ):
                print("resume")
                checkpoint = torch.load(
                    os.path.join(
                        output_dir,
                        f"model{model_num}_P{P}_R_attn{R_attn}R_mlp{R_mlp}_updated.pth",
                    ),
                    map_location="cpu",
                )

                if "model" in checkpoint:
                    model_without_ddp.load_state_dict(checkpoint["model"])
                    pass
                else:
                    ReDecompose_flag = True

                if (
                    not args.eval
                    and "optimizer" in checkpoint
                    and "lr_scheduler" in checkpoint
                    and "epoch" in checkpoint
                ):
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                    args.start_epoch = checkpoint["epoch"] + 1
                    if "scaler" in checkpoint:
                        loss_scaler.load_state_dict(checkpoint["scaler"])
                    if args.model_ema:
                        utils._load_checkpoint_for_ema(
                            model_ema, checkpoint["model_ema"]
                        )

            else:
                ReDecompose_flag = True

            if ReDecompose_flag:
                print("ReDecompose_flag == True")
                absolute_path = os.path.abspath(
                    os.path.join(
                        output_dir, f"model{model_num}_P{P}_R_attn{R_attn}R_mlp{R_mlp}.pth"
                    )
                )

                if utils.is_main_process():
                    utils.save_on_master(
                        model_without_ddp.state_dict(),
                        os.path.join(
                            output_dir,
                            f"model{model_num}_P{P}_R_attn{R_attn}R_mlp{R_mlp}.pth",
                        ),
                    )

                    print("low_rank_decomposition")
                    updated_weights, new_name = low_rank_decompo(
                        original_model=args.pre_trained_model,
                        target_model=absolute_path,
                    )

                    utils.save_on_master(updated_weights, new_name)

                new_name = absolute_path[:-4] + "_updated.pth"

                time.sleep(10)
                while not os.path.exists(new_name):
                    time.sleep(10)

                if utils.is_main_process():
                    write_model_structure(new_name)

                time.sleep(10)
                checkpoint = torch.load(new_name, map_location="cpu")
                model_without_ddp.load_state_dict(checkpoint)

            retrain_config = None
            if args.mode == "retrain" and "RETRAIN" in cfg:
                retrain_config = {
                    "layer_num": cfg.RETRAIN.DEPTH,
                    "embed_dim": [cfg.RETRAIN.EMBED_DIM] * cfg.RETRAIN.DEPTH,
                    "num_heads": cfg.RETRAIN.NUM_HEADS,
                    "mlp_ratio": cfg.RETRAIN.MLP_RATIO,
                }
                for key in retrain_config.keys():
                    print(f"{key}: {retrain_config[key]}")

            if args.eval:
                test_stats, _ = evaluate(
                    data_loader_val,
                    model,
                    device,
                    choices=choices,
                    mode=args.mode,
                    retrain_config=retrain_config,
                )
                print(
                    f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
                )
                return

            print("Start training")
            start_time = time.time()
            max_accuracy = 0.0

            if args.start_epoch == 0:
                if args.output_dir and utils.is_main_process():
                    with open(os.path.join(output_dir, "log.txt"), "a") as f:
                        f.write(
                            f"Training Transformer\nChosen config: {chosen_config[model_num]}\n"
                        )
            elif args.start_epoch < args.stop_epoch:
                if args.output_dir and utils.is_main_process():
                    with open(os.path.join(output_dir, "log.txt"), "a") as f:
                        f.write(
                            f"Training Transformer\nChosen config: {chosen_config[model_num]} continue from epoch {args.start_epoch}\n"
                        )

            for epoch in range(args.start_epoch, args.epochs):
                if epoch >= args.stop_epoch:
                    break

                if args.distributed:
                    data_loader_train.sampler.set_epoch(epoch)

                train_stats = train_one_epoch(
                    model,
                    criterion,
                    data_loader_train,
                    optimizer,
                    device,
                    epoch,
                    loss_scaler,
                    args.clip_grad,
                    model_ema,
                    mixup_fn,
                    amp=args.amp,
                    teacher_model=teacher_model,
                    teach_loss=teacher_loss,
                    choices=choices,
                    mode=args.mode,
                    retrain_config=retrain_config,
                )

                lr_scheduler.step(epoch)
                if args.output_dir:
                    checkpoint_paths = [
                        os.path.join(
                            output_dir,
                            f"model{model_num}_P{P}_R_attn{R_attn}R_mlp{R_mlp}_updated.pth",
                        )
                    ]
                    # checkpoint_paths = [output_dir / "checkpoint.pth"]
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master(
                            {
                                "model": model_without_ddp.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                                "epoch": epoch,
                                # 'model_ema': get_state_dict(model_ema),
                                "scaler": loss_scaler.state_dict(),
                                "args": args,
                            },
                            checkpoint_path,
                        )

                test_stats, parameters = evaluate(
                    data_loader_val,
                    model,
                    device,
                    amp=args.amp,
                    choices=choices,
                    mode=args.mode,
                    retrain_config=retrain_config,
                )

                print(
                    f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
                )
                max_accuracy = max(max_accuracy, test_stats["acc1"])
                print(f"Max accuracy: {max_accuracy:.2f}%")

                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"test_{k}": v for k, v in test_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                    "parameters": parameters,
                }

                if args.output_dir and utils.is_main_process():
                    with open(os.path.join(output_dir, "log.txt"), "a") as f:
                        f.write(json.dumps(log_stats) + "\n")

            args.start_epoch = 0

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print("Training time {}".format(total_time_str))

            sample_num = args.sample_num
            Sample_model(args, sample_num, choices, dataset_val, data_loader_val,
                         device, model_num, P, R_attn, R_mlp, iter, check=True)
            Sample_model(args, sample_num, choices, dataset_val, data_loader_val,
                         device, model_num, P, R_attn, R_mlp, iter, check=True)
            # Sample_model(args, 300, choices, dataset_val, data_loader_val, device, model_num, P, R_attn, R_mlp, iter, check=False)

            if model_num > args.special_num:
                assert None is not None

        chosen_config = patchsize_rankratio_update(
            args, cfg, iter, choices, dataset_val, data_loader_val, device)
        print(f"Updated chosen_config: {chosen_config}")
        print(f"Finished Iteration {iter}")
        print("================================")


def patchsize_rankratio_update(args, cfg, iteration, choices, dataset_val, data_loader_val, device):
    print("================================")
    print(f"Start patchsize_rankratio_update")
    output_dir = Path(args.output_dir)

    results = []
    for iter in range(iteration + 1):
        output_dir = Path(args.output_dir) / f"iteration_{iter}"
        file_path = os.path.join(output_dir, f"results_iteration{iter}.yaml")

        with open(file_path, "r") as f:
            results.append(yaml.safe_load(f))

    while results is None:
        time.sleep(3)

    if utils.is_main_process():
        if len(results) <= 0:
            raise ValueError("results is empty")

        for iter in range(len(results)):
            print(f"results: {results[iter]}")

    X = []
    y = []
    y_delta = []
    model_info = []
    for iter in range(iteration + 1):
        model_num = 0
        for key in results[iter].keys():
            if key == "iteration":
                continue

            if results[iter][key]["u"][-1] == 0:
                model_num += 1
                continue

            x = []
            x.append(int(results[iter][key]["Patch size"]))
            x.append(round(results[iter][key]["Rank ratio attn"] / 100, 2))
            x.append(round(results[iter][key]["Rank ratio mlp"] / 100, 2))
            x.append(1)
            X.append(x)

            y.append(results[iter][key]["u"][-1])
            y_delta.append(results[iter][key]["delta_u"][-1])

            info = {}
            info["iter"] = iter
            info["model_num"] = model_num
            info["P"] = x[0]
            info["R_attn"] = x[1]
            info["R_mlp"] = x[2]
            model_info.append(info)

            model_num += 1

    last_iter_model_num = 0
    for i in range(len(model_info)):
        print(f"{i}th model: {model_info[i]}")
        if model_info[i]["iter"] == iteration:
            last_iter_model_num += 1

    print(f"last_iter_model_num: {last_iter_model_num}")

    X = np.array(X)
    y = np.array(y)
    y_delta = np.array(y_delta)
    print("X.shape:", X.shape)
    print("y.shape:", y.shape)
    print("y_delta.shape:", y_delta.shape)

    print("X:", X)
    print("y:", y)
    print("y_delta:", y_delta)

    # 创建KDTree
    normalized_X = np.copy(X)
    normalized_X[:, 0] = (normalized_X[:, 0] - min(cfg.SUPERNET.PATCH_SIZE)) / (
        max(cfg.SUPERNET.PATCH_SIZE) - min(cfg.SUPERNET.PATCH_SIZE)) + 1e-8
    # 0 1 0.33
    normalized_X[:, 0] = normalized_X[:, 0] / 2

    normalized_X[:, 1] = (normalized_X[:, 1] - min(cfg.SUPERNET.RANK_RATIO)) / (
        max(cfg.SUPERNET.RANK_RATIO) - min(cfg.SUPERNET.RANK_RATIO)) + 1e-8
    # 0 1 0.033 * 5
    normalized_X[:, 2] = (normalized_X[:, 2] - min(cfg.SUPERNET.RANK_RATIO)) / (
        max(cfg.SUPERNET.RANK_RATIO) - min(cfg.SUPERNET.RANK_RATIO)) + 1e-8
    tree = cKDTree(normalized_X)
    k = 8

    # 对每个点找出最近的8个邻居（包括自己）
    neighbors_idx = tree.query(normalized_X, k=k)[1]  # 返回索引，k=6因为包括点本身
    print(
        f"Finished Loading Data, Start updating the last {last_iter_model_num} models")

    X_update = np.copy(X[-last_iter_model_num:, :3])
    y_update = []

    # 对每个点及其邻居进行线性拟合并更新点
    for i, idx in enumerate(neighbors_idx[-last_iter_model_num:]):
        print("================================")
        print(f"Point {i}")
        print(f"Current Point {i}: {X_update[i]}")

        # 获取当前点及其邻居的X和y值
        X_neighbors = X[idx]
        y_neighbors = y[idx]
        y_delta_neighbors = y_delta[idx]

        flag = False
        irls_count = 0
        while (not flag):
            print(f"IRLS Iteration {irls_count}")
            beta, weights = irls(X_neighbors, y_neighbors, cfg)
            print(f"beta: {beta}")
            print(f"weights: {weights}")
            print(f"y_delta_neighbors: {y_delta_neighbors}")

            constant = 1e-4
            products = weights * y_delta_neighbors
            print(f"products: {products}")
            products = products < constant
            resample_ids = []

            for m in range(k):
                if products[m]:
                    continue
                else:
                    resample_ids.append(idx[m])

            print(f"resample_ids: {resample_ids}")
            for id in resample_ids:
                print(f"Resample Point {id}: {X[id]}")

            sample_num = args.sample_num
            if len(resample_ids) > 0:
                for id in resample_ids:

                    info = model_info[id]
                    iter_4_sample = info["iter"]
                    model_num_4_sample = info["model_num"]
                    P = int(info["P"])
                    R_attn = int(info["R_attn"] * 100)
                    R_mlp = int(info["R_mlp"] * 100)

                    y[id], y_delta[id] = Sample_model(
                        args, sample_num, choices, dataset_val, data_loader_val, device, model_num_4_sample, P, R_attn, R_mlp, iter_4_sample)
                y_neighbors = y[idx]
                y_delta_neighbors = y_delta[idx]
            else:
                flag = True

        # 计算更新方向并更新点
        # 注意：这里假设X[i]是需要更新的点
        P_smallest = min(cfg.SUPERNET.PATCH_SIZE)
        P_largest = max(cfg.SUPERNET.PATCH_SIZE)
        P_range = P_largest - P_smallest

        R_smallest = min(cfg.SUPERNET.RANK_RATIO)
        R_largest = max(cfg.SUPERNET.RANK_RATIO)
        R_range = R_largest - R_smallest
        print(f"P: {P_smallest} - {P_largest}")
        print(f"R: {R_smallest} - {R_largest}")

        y_orig = beta[0] * (X_update[i, 0] - P_smallest) / (P_range)
        y_orig += beta[1] * (X_update[i, 1] - R_smallest) / (R_range)
        y_orig += beta[2] * (X_update[i, 2] - R_smallest) / (R_range)
        y_orig += beta[3]
        print(f"Original Point {i}: {X_update[i]} Acc by irls: {y_orig}")

        update_direction = beta[:3]
        step = 2 * np.around(update_direction[0] / 0.05)
        step = np.clip(step, -2, 2)
        X_update[i, 0] = int(step + X_update[i, 0])  # 更新点
        X_update[i, 0] = int(np.clip(X_update[i, 0], P_smallest, P_largest))

        step = 0.02 * np.around(update_direction[1:] / 0.005).astype(int)
        step = np.clip(step, -0.06, 0.06)
        X_update[i, 1:] += step  # 更新点
        X_update[i, 1:] = np.around(
            np.clip(X_update[i, 1:], R_smallest, R_largest), 2)  # 保证在0-1之间

        y_pred = beta[0] * (X_update[i, 0] - P_smallest) / (P_range)
        y_pred += beta[1] * (X_update[i, 1] - R_smallest) / (R_range)
        y_pred += beta[2] * (X_update[i, 2] - R_smallest) / (R_range)
        y_pred += beta[3]
        y_update.append(y_pred)

        # 输出更新后的点及其预测值
        print(f"Updated Point {i}: {X_update[i]} Expected Acc: {y_update[i]}")
        print("================================")

    print("================================")
    for i in range(X_update.shape[0]):
        print(
            f"Original Point {i}: {X[-(last_iter_model_num - i), :3]} Real Acc: {y[-(last_iter_model_num - i)]}")
        print(
            f"\t Updated Point {i}: {X_update[i]} Expected Acc: {y_update[i]}")
    print("================================")

    X_update = X_update.tolist()
    X = X[:, :3].tolist()
    for j in range(len(X)):
        X[j][0] = int(X[j][0])
        X[j][1] = round(X[j][1], 2)
        X[j][2] = round(X[j][2], 2)

    for l in range(len(X_update)):
        X_update[l][0] = int(X_update[l][0])
        X_update[l][1] = round(X_update[l][1], 2)
        X_update[l][2] = round(X_update[l][2], 2)

    X_update2 = []
    y_update2 = []
    print("================================")
    print("Removing repeated points:")
    for l in range(len(X_update)):
        if X_update[l] not in X and X_update[l] not in X_update2:
            if check_valid_subnet_num(args, X_update[l][0], X_update[l][1], X_update[l][2], cfg, choices, sample_max_iter=1e4):
                X_update2.append(X_update[l])
                y_update2.append(y_update[l])
            else:
                print(f"Point {X_update[l]} excceds the memory limit.")
        else:
            print(f"Point {X_update[l]} is already in X")

    for i in range(len(X_update2)):
        print(
            f"Updated Point {i}: {X_update2[i]} Expected Acc: {y_update2[i]}")
    print("================================")

    Next_model_num = 0
    if iteration == 0:
        Next_model_num = 8
    else:
        Next_model_num = 6
    sorted_indices = np.argsort(np.array(y_update2))[::-1]
    X_update2_sorted = [X_update2[i] for i in sorted_indices]
    y_update2_sorted = [y_update2[i] for i in sorted_indices]

    print("================================")
    print("Sorted points:")
    for i in range(len(X_update2_sorted)):
        print(
            f"Sorted Point {i}: {X_update2_sorted[i]} Expected Acc: {y_update2_sorted[i]}")
    print("================================")

    X_update2_selected = X_update2_sorted[:Next_model_num]
    y_update2_selected = y_update2_sorted[:Next_model_num]

    print("================================")
    print(f"Select the best {Next_model_num} points")
    print("Selected points:")
    for i in range(len(X_update2_selected)):
        print(
            f"Updated Point {i}: {X_update2_selected[i]} Expected Acc: {y_update2_selected[i]}")
    print("================================")
    return X_update2_selected


def irls(X, y, cfg, max_iter=1000, tol=1e-5):
    """
    迭代重加权最小二乘法 (IRLS)

    参数:
    X -- 输入变量矩阵，形状为 (n_samples, n_features)
    y -- 目标变量向量，形状为 (n_samples,)
    max_iter -- 最大迭代次数
    tol -- 收敛阈值

    y = Wx + b

    返回:
    beta -- 参数估计值，形状为 (n_features,)
    weights -- 重加权因子，形状为 (n_samples,)
    """
    print("================================")
    print("Start IRLS")
    print("X:", X)
    print("y:", y)
    n_samples, n_features = X.shape
    # 初始化参数估计值
    beta = np.zeros(n_features)

    # 归一化X
    P_smallest = min(cfg.SUPERNET.PATCH_SIZE)
    P_range = max(cfg.SUPERNET.PATCH_SIZE) - P_smallest

    R_smallest = min(cfg.SUPERNET.RANK_RATIO)
    R_range = max(cfg.SUPERNET.RANK_RATIO) - R_smallest

    X_irls = np.copy(X)
    X_irls[:, 0] = (X_irls[:, 0] - P_smallest) / (P_range) + 1e-8
    X_irls[:, 1:3] = (X_irls[:, 1:3] - R_smallest) / (R_range) + 1e-8
    print("Normalized X in irls:", X_irls)

    for iteration in range(max_iter):
        # 计算预测值
        y_pred = X_irls @ beta
        # 计算残差
        residuals = y - y_pred

        # 计算权重，这里是一个示例，具体权重的计算取决于问题
        # 对于逻辑回归，权重会根据当前的预测值来计算
        # weights = 1 / (1 + np.abs(residuals))
        p = 1
        weights = np.abs(residuals) ** ((p - 2) / 2)
        weights = weights / np.sum(weights)

        # 构建加权设计矩阵
        W = np.diag(weights)
        # 更新参数估计值，使用加权最小二乘公式
        beta_new = np.linalg.inv(
            X_irls.T @ W @ X_irls + 1e-6 * np.eye(X_irls.shape[1])) @ (X_irls.T @ W @ y)

        # 检查收敛
        if np.linalg.norm(beta_new - beta) < tol or iteration == max_iter - 1:
            print(f"Converged at iteration {iteration}")
            break
        beta = np.copy(beta_new)

    print("================================")
    return beta, weights


def Sample_model(args, sample_num, choices, dataset_val, data_loader_val, device, model_num, P, R_attn, R_mlp, iter, check=False):

    def Update_results(args, results: dict, index: str, sample_num, mean_acc, min_acc, max_acc):
        if len(results[index]["Mean_acc"]) > 0:
            mean_acc = (results[index]["Sample num"] * results[index]["Mean_acc"][-1] +
                        sample_num * mean_acc) / (results[index]["Sample num"] + sample_num)
            results[index]["Mean_acc"].append(mean_acc)
        else:
            results[index]["Mean_acc"].append(mean_acc)

        if len(results[index]["Min_acc"]) > 0:
            results[index]["Min_acc"].append(
                min(min_acc, results[index]["Min_acc"][-1]))
        else:
            results[index]["Min_acc"].append(min_acc)

        if len(results[index]["Max_acc"]) > 0:
            results[index]["Max_acc"].append(
                max(max_acc, results[index]["Max_acc"][-1]))
        else:
            results[index]["Max_acc"].append(max_acc)

        if mean_acc == 0 and results[index]["Max_acc"][-1] == 0 and results[index]["Min_acc"][-1] == 0:
            u = 0
        else:
            u = math.sqrt(-math.log(args.upper_bound_prob) * (
                (results[index]["Max_acc"][-1] - results[index]["Min_acc"][-1]) ** 2) / 2)
            u = mean_acc + u
        results[index]["u"].append(u)

        if len(results[index]["u"]) >= 2:
            delta_u = abs(results[index]["u"][-1] - results[index]["u"][-2])
            results[index]["delta_u"].append(delta_u)

        results[index]["Sample num"] = int(
            results[index]["Sample num"]) + sample_num

        print("-------------------------------------------")
        print(f"Updated results for index: {index}")
        print("Results:", results[index])
        print("-------------------------------------------")

        return results

    def Sample(args, sample_num, choices, model_without_ddp, data_loader_val, model, device):
        print("================================")
        print("Start Sampling")

        min_acc = 100
        max_acc = 0

        acc = np.array([])
        count = 0
        max_count = 100000
        while len(acc) < (sample_num) and count < max_count:
            config = sample_configs(choices=choices)
            max_memory = model_without_ddp.get_maximum_memory(config)
            if float(max_memory >= args.limit_memory):
                count += 1
                continue
            print(f"Sampling {len(acc)}th config")
            print(f"Count: {count}")
            test_stats, _ = evaluate(
                data_loader_val,
                model,
                device,
                choices=None,
                mode="retrain_config",
                retrain_config=config,
            )
            count = 0
            print(
                f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
            )

            acc = np.append(acc, test_stats["acc1"] / 100)
            min_acc = min(min_acc, test_stats["acc1"] / 100)
            max_acc = max(max_acc, test_stats["acc1"] / 100)

        if len(acc) == 0:
            return sample_num, 0, 0, 0

        mean_acc = np.mean(acc).item()

        print("Finished Sampling")
        print("================================")
        return sample_num, mean_acc, min_acc, max_acc

    results = None

    output_dir = Path(args.output_dir) / f"iteration_{iter}"
    file_path = os.path.join(output_dir, f"results_iteration{iter}.yaml")

    if os.path.exists(file_path):
        while (results is None):
            with open(file_path, "r") as f:
                results = yaml.safe_load(f)

    if results is None:
        results = {}
        results['iteration'] = iter

    index = f"model{model_num}P{P}_R_attn{R_attn}R_mlp{R_mlp}"

    print(f"Start calculating score of iteration {iter} {index}.")
    score_time = time.time()

    if check:
        if index in results.keys():
            if "Sample num" in results[index].keys():
                if results[index]["Sample num"] >= 20:
                    sampled_num = results[index]["Sample num"]
                    print(
                        f"Sample num of {index} is {sampled_num}. Stop sampling")
                    return

    if index not in results.keys():
        results[index] = {}
        results[index]["Model num"] = model_num
        results[index]["Patch size"] = int(P)
        results[index]["Rank ratio attn"] = R_attn
        results[index]["Rank ratio mlp"] = R_mlp
        results[index]["Mean_acc"] = []
        results[index]["Min_acc"] = []
        results[index]["Max_acc"] = []
        results[index]["u"] = []
        results[index]["delta_u"] = []
        results[index]["Sample num"] = 0

    model = Vision_TransformerSuper(
        img_size=args.input_size,
        patch_size=int(P),
        embed_dim=cfg.SUPERNET.EMBED_DIM,
        depth=cfg.SUPERNET.DEPTH,
        num_heads=cfg.SUPERNET.NUM_HEADS,
        mlp_ratio=cfg.SUPERNET.MLP_RATIO,
        qkv_bias=True,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        gp=args.gp,
        num_classes=args.nb_classes,
        max_relative_position=args.max_relative_position,
        relative_position=args.relative_position,
        change_qkv=args.change_qkv,
        abs_pos=not args.no_abs_pos,
        rank_ratio_attn=round(R_attn / 100, 2),
        rank_ratio_mlp=round(R_mlp / 100, 2),
    )

    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    file_path = os.path.join(
        output_dir, f"model{model_num}_P{P}_R_attn{R_attn}R_mlp{R_mlp}_updated.pth")
    print(f"resume from {file_path}")
    checkpoint = torch.load(file_path, map_location="cpu")
    model_without_ddp.load_state_dict(checkpoint["model"])

    sample_num, mean_acc, min_acc, max_acc = Sample(
        args, sample_num, choices, model_without_ddp, data_loader_val, model, device)
    results = Update_results(args, results, index,
                             sample_num, mean_acc, min_acc, max_acc)

    if utils.is_main_process():
        file_path = os.path.join(output_dir, f"results_iteration{iter}.yaml")
        with open(os.path.join(file_path), "w") as f:
            yaml.dump(results, f, default_flow_style=False)

        file_path = os.path.join(output_dir, f"log.txt")
        with open(os.path.join(file_path), "a") as f:
            f.write(f"Iteration{iter}: {index} sampled and updated.\n")

    score_time = time.time() - score_time
    score_time_str = str(datetime.timedelta(seconds=int(score_time)))
    print("Score time {}".format(score_time_str))

    if len(results[index]["u"]) >= 2:
        print(
            f"u: {results[index]['u'][-1]}, delta_u: {results[index]['delta_u'][-1]}")
        return results[index]["u"][-1], results[index]["delta_u"][-1]


def check_valid_subnet_num(args, patch_size, r_attn, r_mlp, cfg, choices, sample_max_iter=1000, valid_subnet_num=5) -> bool:
    print("================================")
    print(
        f"Start checking memory. Patch size: {patch_size}, Rank ratio attn: {r_attn}, Rank ratio mlp: {r_mlp}")
    model = Vision_TransformerSuper(
        img_size=args.input_size,
        patch_size=int(patch_size),
        embed_dim=cfg.SUPERNET.EMBED_DIM,
        depth=cfg.SUPERNET.DEPTH,
        num_heads=cfg.SUPERNET.NUM_HEADS,
        mlp_ratio=cfg.SUPERNET.MLP_RATIO,
        qkv_bias=True,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        gp=args.gp,
        num_classes=args.nb_classes,
        max_relative_position=args.max_relative_position,
        relative_position=args.relative_position,
        change_qkv=args.change_qkv,
        abs_pos=not args.no_abs_pos,
        rank_ratio_attn=r_attn,
        rank_ratio_mlp=r_mlp,
    )
    count = 0
    subnet_count = 0
    while count < sample_max_iter and subnet_count < valid_subnet_num:
        config = sample_configs(choices=choices)
        max_memory = model.get_maximum_memory(config)
        if float(max_memory <= args.limit_memory):
            subnet_count += 1

        count += 1

    if subnet_count >= valid_subnet_num:
        print(
            f"Patch size: {patch_size}, Rank ratio attn: {r_attn}, Rank ratio mlp: {r_mlp} does not exceed the memory limit")
        print("================================")
        return True
    else:
        print(
            f"Patch size: {patch_size}, Rank ratio attn: {r_attn}, Rank ratio mlp: {r_mlp} exceeds the memory limit")
        print("================================")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "AutoFormer training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
