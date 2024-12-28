import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from model.module.embedding_super import PatchembedSuper
from model.module.layernorm_super import LayerNormSuper
from model.module.Linear_super import LinearSuper
from model.module.multihead_super import AttentionSuper
from model.utils import DropPath, trunc_normal_


def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    GELU 激活函數
    """
    if hasattr(torch.nn.functional, "gelu"):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Vision_TransformerSuper(nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        pre_norm=True,
        scale=False,
        gp=False,
        relative_position=False,
        change_qkv=False,
        abs_pos=True,
        max_relative_position=14,
        rank_ratio_attn=0.8,
        rank_ratio_mlp=0.8,
    ):
        super().__init__()
        # the configs of super arch
        self.super_embed_dim = embed_dim
        # self.super_embed_dim = args.embed_dim
        self.super_mlp_ratio = mlp_ratio
        self.super_layer_num = depth
        self.super_num_heads = num_heads
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate
        self.num_classes = num_classes
        self.pre_norm = pre_norm
        self.scale = scale
        
        self.rank_ratio_attn = rank_ratio_attn
        self.rank_ratio_mlp = rank_ratio_mlp

        self.patch_embed_super = PatchembedSuper(img_size=img_size, patch_size=patch_size,
                                                 in_chans=in_chans, embed_dim=embed_dim)

        self.gp = gp

        # configs for the sampled subTransformer
        self.sample_patch_size = None
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        self.sample_dropout = None
        self.sample_output_dim = None

        self.blocks = nn.ModuleList()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        for i in range(depth):
            self.blocks.append(
                TransformerEncoderLayer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    pre_norm=pre_norm,
                    scale=self.scale,
                    change_qkv=change_qkv,
                    relative_position=relative_position,
                    max_relative_position=max_relative_position,
                    rank_ratio_attn=rank_ratio_attn,
                    rank_ratio_mlp=rank_ratio_mlp,
                )
            )
        num_patches = self.patch_embed_super.num_patches
        
        self.abs_pos = abs_pos
        if self.abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)
        if self.pre_norm:
            self.norm = LayerNormSuper(super_embed_dim=embed_dim)

        # classifier head
        self.head = (
            LinearSuper(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "rel_pos_embed"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def set_sample_config(self, config: dict):
        self.sample_embed_dim = config["embed_dim"]
        self.sample_mlp_ratio = config["mlp_ratio"]
        self.sample_layer_num = config["layer_num"]
        self.sample_num_heads = config["num_heads"]
        self.sample_dropout = calc_dropout(
            self.super_dropout, self.sample_embed_dim[0], self.super_embed_dim
        )

        self.patch_embed_super.set_sample_config(self.sample_embed_dim[0])

        self.sample_output_dim = [out_dim for out_dim in self.sample_embed_dim[1:]] + [
            self.sample_embed_dim[-1]
        ]
        for i, blocks in enumerate(self.blocks):
            # not exceed sample layer number
            if i < self.sample_layer_num:
                sample_dropout = calc_dropout(
                    self.super_dropout, self.sample_embed_dim[i], self.super_embed_dim
                )
                sample_attn_dropout = calc_dropout(
                    self.super_attn_dropout,
                    self.sample_embed_dim[i],
                    self.super_embed_dim,
                )
                blocks.set_sample_config(
                    is_identity_layer=False,
                    sample_embed_dim=self.sample_embed_dim[i],
                    sample_mlp_ratio=self.sample_mlp_ratio[i],
                    sample_num_heads=self.sample_num_heads[i],
                    sample_dropout=sample_dropout,
                    sample_out_dim=self.sample_output_dim[i],
                    sample_attn_dropout=sample_attn_dropout,
                )
            # exceeds sample layer number
            else:
                blocks.set_sample_config(is_identity_layer=True)
        if self.pre_norm:
            self.norm.set_sample_config(self.sample_embed_dim[-1])
        self.head.set_sample_config(self.sample_embed_dim[-1], self.num_classes)

    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_modules():
            if hasattr(module, "calc_sampled_param_num"):
                if (
                    name.split(".")[0] == "blocks"
                    and int(name.split(".")[1]) >= config["layer_num"]
                ):
                    continue
                numels.append(module.calc_sampled_param_num())

        return sum(numels) + self.sample_embed_dim[0] * (
            2 + self.patch_embed_super.num_patches
        )

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.patch_embed_super.get_complexity(sequence_length)
        total_flops += (
            np.prod(self.pos_embed[..., : self.sample_embed_dim[0]].size()) / 2.0
        )
        for blk in self.blocks:
            total_flops += blk.get_complexity(sequence_length + 1)
        total_flops += self.head.get_complexity(sequence_length + 1)
        return total_flops

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed_super(x)

        cls_tokens = self.cls_token[..., : self.sample_embed_dim[0]].expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.abs_pos:
            x = x + self.pos_embed[..., : self.sample_embed_dim[0]]

        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        for blk in self.blocks:
            x = blk(x)

        if self.pre_norm:
            x = self.norm(x)

        if self.gp:
            return torch.mean(x[:, 1:], dim=1)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def get_maximum_memory(self, config):
        """
        Get the maximum memory usage of the model
        """
        max_memory = 0
        self.set_sample_config(config)
        T = self.patch_embed_super.num_patches + 1
        # get the maximum memory usage of the model
        for i, blk in enumerate(self.blocks):
            if i >= config["layer_num"]:
                break
            E = max(blk.sample_embed_dim, blk.sample_out_dim)
            M = blk.sample_mlp_ratio
            R_mlp = blk.rank_ratio_mlp
            memory_mlp = E * T * M + int(E * R_mlp) * T  + E * M * int(E * R_mlp)
            
            E = blk.sample_embed_dim
            R_attn = blk.rank_ratio_attn
            QKV = blk.sample_num_heads_this_layer * 64 * 3
            memory_attn = QKV * T + int(E * R_attn) * T + QKV * int(E * R_attn)
            
            memory = max(memory_mlp, memory_attn)
            max_memory = max(max_memory, memory)

        return max_memory


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments which
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        dropout=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        pre_norm=True,
        scale=False,
        relative_position=False,
        change_qkv=False,
        max_relative_position=14,
        rank_ratio_attn=0.8,
        rank_ratio_mlp=0.8,
    ):
        super().__init__()

        # the configs of super arch of the encoder, three dimension [embed_dim, mlp_ratio, and num_heads]
        self.super_embed_dim = dim
        self.super_mlp_ratio = mlp_ratio
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.super_num_heads = num_heads
        self.normalize_before = pre_norm
        self.super_dropout = attn_drop
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale = scale
        self.relative_position = relative_position
        self.rank_ratio_attn = rank_ratio_attn
        self.rank_ratio_mlp = rank_ratio_mlp

        # self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_heads_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_attn_dropout = None

        self.is_identity_layer = None
        self.attn = AttentionSuper(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=dropout,
            scale=self.scale,
            relative_position=self.relative_position,
            change_qkv=change_qkv,
            max_relative_position=max_relative_position,
            rank_ratio_attn=rank_ratio_attn,
        )

        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim)
        # self.dropout = dropout
        self.activation_fn = gelu
        # self.normalize_before = args.encoder_normalize_before

        # self.fc1 = LinearSuper(
        #     super_in_dim=self.super_embed_dim,
        #     super_out_dim=self.super_ffn_embed_dim_this_layer,
        # )

        # self.fc2 = LinearSuper(
        #     super_in_dim=self.super_ffn_embed_dim_this_layer,
        #     super_out_dim=self.super_embed_dim,
        # )
        
        self.fc11 = LinearSuper(
            super_in_dim=self.super_embed_dim,
            super_out_dim=int(self.super_embed_dim * rank_ratio_mlp),
            bias=False,
        )

        # self.fc12 = LinearSuper(
        #     super_in_dim=int(self.super_embed_dim * rank_ratio),
        #     super_out_dim=int(self.super_embed_dim * rank_ratio),
        #     bias=False,
        # )

        self.fc13 = LinearSuper(
            super_in_dim=int(self.super_embed_dim * rank_ratio_mlp),
            super_out_dim=self.super_ffn_embed_dim_this_layer,
        )

        self.fc21 = LinearSuper(
            super_in_dim=self.super_ffn_embed_dim_this_layer,
            super_out_dim=int(self.super_embed_dim * rank_ratio_mlp),
            bias=False,
        )

        # self.fc22 = LinearSuper(
        #     super_in_dim=int(self.super_embed_dim * rank_ratio),
        #     super_out_dim=int(self.super_embed_dim * rank_ratio),
        #     bias=False,
        # )

        self.fc23 = LinearSuper(
            super_in_dim=int(self.super_embed_dim * rank_ratio_mlp),
            super_out_dim=self.super_embed_dim,
        )

    def set_sample_config(
        self,
        is_identity_layer,
        sample_embed_dim=None,
        sample_mlp_ratio=None,
        sample_num_heads=None,
        sample_dropout=None,
        sample_attn_dropout=None,
        sample_out_dim=None,
    ):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_ffn_embed_dim_this_layer = int(sample_embed_dim * sample_mlp_ratio)
        self.sample_num_heads_this_layer = sample_num_heads
        # self.sample_rank_ratio = sample_rank_ratio

        self.sample_dropout = sample_dropout
        self.sample_attn_dropout = sample_attn_dropout
        self.attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        self.attn.set_sample_config(
            sample_q_embed_dim=self.sample_num_heads_this_layer * 64,
            sample_num_heads=self.sample_num_heads_this_layer,
            sample_in_embed_dim=self.sample_embed_dim,
        )

        # self.fc1.set_sample_config(
        #             sample_in_dim=self.sample_embed_dim,
        #             sample_out_dim=self.sample_ffn_embed_dim_this_layer,
        #         )

        # self.fc2.set_sample_config(
        #             sample_in_dim=self.sample_ffn_embed_dim_this_layer,
        #             sample_out_dim=self.sample_out_dim,
        #         )

        self.fc11.set_sample_config(
            sample_in_dim=self.sample_embed_dim,
            sample_out_dim=int(self.sample_embed_dim * self.rank_ratio_mlp),
        )
        
        self.fc13.set_sample_config(
            sample_in_dim=int(self.sample_embed_dim * self.rank_ratio_mlp),
            sample_out_dim=self.sample_ffn_embed_dim_this_layer,
        )

        self.fc21.set_sample_config(
            sample_in_dim=self.sample_ffn_embed_dim_this_layer,
            sample_out_dim=int(self.sample_embed_dim * self.rank_ratio_mlp),
        )
        
        self.fc23.set_sample_config(
            sample_in_dim=int(self.sample_embed_dim * self.rank_ratio_mlp),
            sample_out_dim=self.sample_out_dim,
        )

        self.ffn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, patch_num , sample_embed_dim)`

        Returns:
            encoded output of shape `(batch, patch_num, sample_embed_dim)`
        """
        if self.is_identity_layer:
            return x

        residual = x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x)
        x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)

        x = self.fc11(x)
        # x = self.fc12(x)
        x = self.fc13(x)

        x = self.activation_fn(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        x = self.fc21(x)
        # x = self.fc22(x)
        x = self.fc23(x)

        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        if self.scale:
            x = x * (self.super_mlp_ratio / self.sample_mlp_ratio)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)

        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.is_identity_layer:
            return total_flops
        total_flops += self.attn_layer_norm.get_complexity(sequence_length + 1)
        total_flops += self.attn.get_complexity(sequence_length + 1)
        total_flops += self.ffn_layer_norm.get_complexity(sequence_length + 1)
        total_flops += self.fc11.get_complexity(sequence_length + 1)
        # total_flops += self.fc12.get_complexity(sequence_length + 1)
        total_flops += self.fc13.get_complexity(sequence_length + 1)
        total_flops += self.fc21.get_complexity(sequence_length + 1)
        # total_flops += self.fc22.get_complexity(sequence_length + 1)
        total_flops += self.fc23.get_complexity(sequence_length + 1)
        return total_flops

    def calc_sampled_param_num(self):
        num = 0
        num += self.fc11.calc_sampled_param_num()
        # num += self.fc12.calc_sampled_param_num()
        num += self.fc13.calc_sampled_param_num()
        num += self.fc21.calc_sampled_param_num()
        # num += self.fc22.calc_sampled_param_num()
        num += self.fc23.calc_sampled_param_num()
        return num


def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim
