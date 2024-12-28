import math

import numpy as np
import torch
import torch.nn as nn


def svd_decompo(
    weights1: dict,
    weights2: dict,
    old_key_w,
    old_key_b,
    new_key_w1,
    new_key_b1,
    new_key_w2,
    new_key_b2,
    new_key_w3,
    new_key_b3,
):
    torch.manual_seed(0)
    if old_key_w not in weights1["model"]:
        print(f"{old_key_w} not in weights1")
        return weights2

    original_weight = weights1["model"][old_key_w]

    assert weights2[new_key_w1].size(0) == weights2[new_key_w3].size(1)

    k = weights2[new_key_w1].size(0)
    U2, S2, V2 = torch.svd_lowrank(original_weight, q=k)

    U_new = U2
    S_new = S2
    V_new = V2

    weights2[new_key_w1].copy_(V_new.t())
    weights2[new_key_w3].copy_(torch.matmul(U_new, torch.diag(S_new))) 
    weights2[new_key_b3].copy_(weights1["model"][old_key_b])

    print(f"weights2 {new_key_w1} updated")
    
    return weights2


def low_rank_decompo(
    original_model: str = "supernet-tiny.pth",
    target_model: str = "target_model.pth",
):
    print(f"original_model: {original_model}")
    print(f"target_model: {target_model}")

    weights1 = torch.load(original_model, map_location="cpu")
    weights2 = torch.load(target_model, map_location="cpu")

    for key in weights2:
        if key.startswith("patch_embed_super") and key.endswith("weight"):
            weights2[key].copy_(nn.init.kaiming_uniform_(weights2[key], a=math.sqrt(5)))
            continue

        elif key.startswith("patch_embed_super") and key.endswith("bias"):
            weights2[key].copy_(nn.init.uniform_(weights2[key], 0, 5))
            continue

        elif key.startswith("pos_embed_dict") or key.startswith("pos_embed"):
            weights2[key].copy_(nn.init.kaiming_uniform_(weights2[key], a=math.sqrt(5)))
            continue

        elif key not in weights1["model"]:
            print(f"{key} not in weights1")
            continue
        
        else:
            if weights1["model"][key].size() != weights2[key].size():
                print(f"size of {key} not equal")
                
            else:
                weights2[key].copy_(weights1["model"][key])
                print(f"copy {key} from weights1 to weights2")


    for i in range(16):
        key_weight = f"blocks.{i}.attn.qkv.weight"
        key_bias = f"blocks.{i}.attn.qkv.bias"

        key1 = f"blocks.{i}.attn.qkv1.weight"
        key1_bias = f"blocks.{i}.attn.qkv1.bias"
        key2 = f"blocks.{i}.attn.qkv2.weight"
        key2_bias = f"blocks.{i}.attn.qkv2.bias"
        key3 = f"blocks.{i}.attn.qkv3.weight"
        key3_bias = f"blocks.{i}.attn.qkv3.bias"

        weights2 = svd_decompo(
            weights1,
            weights2,
            key_weight,
            key_bias,
            key1,
            key1_bias,
            key2,
            key2_bias,
            key3,
            key3_bias,
        )
        
        for j in range(1, 3):
            key = f"blocks.{i}.fc{j}.weight"
            key_bias = f"blocks.{i}.fc{j}.bias"

            key1 = f"blocks.{i}.fc{j}1.weight"
            key1_bias = f"blocks.{i}.fc{j}1.bias"

            key2 = f"blocks.{i}.fc{j}2.weight"
            key2_bias = f"blocks.{i}.fc{j}2.bias"

            key3 = f"blocks.{i}.fc{j}3.weight"
            key3_bias = f"blocks.{i}.fc{j}3.bias"

            weights2 = svd_decompo(
                weights1,
                weights2,
                key,
                key_bias,
                key1,
                key1_bias,
                key2,
                key2_bias,
                key3,
                key3_bias,
            )

    new_name = target_model[:-4] + "_updated.pth"
    
    return weights2, new_name


# if __name__ == "__main__":
#     weights, new_name = low_rank_decompo(
#         original_model="/home/ubuntu/louis_crq/AutoFormer_MCU_louis_copy/structure/supernet-tiny.pth",
#         target_model="/home/ubuntu/louis_crq/AutoFormer_MCU_louis_copy/model.pth",
#         rank_ratio=[0.7],
#     )

#     torch.save(weights, new_name)

# python /home/ubuntu/louis_crq/AutoFormer_MCU_louis_copy/structure/update_structure.py
