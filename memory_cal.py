from model.module.embedding_super import PatchembedSuper

memory_limit = 256*1024
print(f"memory_limit: {memory_limit}")

# P = [16, 18, 20, 22, 24, 26]
P = [20, 22, 24, 26]

for patch_size in P:
    T = PatchembedSuper(patch_size=patch_size)
    T = T.num_patches  # 【196， 169， 144， 121， 100】
    E = 224
    M = 4
    QKV = 4 * 3 * 64

    # print(f"patch_size: {patch_size}, T: {T}")

    for R in range(65, 96):
        R_mlp = R / 100
        memory = E * T * M + int(E * R_mlp) * T + E * M * int(E * R_mlp)

        if memory > memory_limit:
            # R_mlp -= 0.01
            # memory = E * T * M + int(E * R_mlp) * T  + E * M * int(E * R_mlp)
            print(
                f"patch_size: {patch_size}, R_mlp: {R_mlp}, memory: {memory}")
            break

    for R in range(65, 96):
        R_attn = R / 100
        memory = QKV * T + int(E * R_attn) * T + QKV * int(E * R_attn)

        if memory > memory_limit:
            # R_attn -= 0.01
            # memory = QKV * T + int(E * R_attn) * T + QKV * int(E * R_attn)
            print(
                f"patch_size: {patch_size}, R_attn: {R_attn}, memory: {memory}")
            break

# T = 145
# R = 0.85
# memory = E * T * M + int(E * R) * T  + E * M * int(E * R)
# print(f"patch_size: {20}, R_mlp: {R}, memory: {memory}")

# memory = QKV * T + int(E * R) * T + QKV * int(E * R)
# print(f"patch_size: {20}, R_attn: {R}, memory: {memory}")
