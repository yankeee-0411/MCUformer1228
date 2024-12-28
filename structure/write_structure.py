import torch

def write_model_structure(pthname: str):
    checkpoint = torch.load(pthname, map_location=torch.device("cpu"))

    with open(pthname[:-4] + ".txt", "w") as file:
        for key, value in checkpoint.items():
            file.write(f"Key: {key}\n")
            file.write(f"Type: {type(value)}\n")
            if torch.is_tensor(value):
                file.write(f"Size: {value.size()}\n")
                # file.write(f"Value: {value}")
            file.write("\n")
    
    

# checkpoint = torch.load(
#     "/home/ubuntu/louis_crq/AutoFormer_MCU_louis/model_with_svd.pth", map_location=torch.device("cpu")
# )

# with open("/home/ubuntu/louis_crq/AutoFormer_MCU_louis/model_with_svd.txt", "w") as file:
#     for key, value in checkpoint.items():
#         file.write(f"Key: {key}\n")
#         file.write(f"Type: {type(value)}\n")
#         if torch.is_tensor(value):
#             file.write(f"Size: {value.size()}\n")
#             file.write(f"Value: {value}")
#         file.write("\n")

# name = "/home/ubuntu/louis_crq/AutoFormer_MCU_louis_copy/model"

# checkpoint = torch.load("/home/ubuntu/louis_crq/AutoFormer_MCU_louis0612/output_louis0/iteration_1/model0_P16_R_attn95R_mlp80_updated.pth", map_location=torch.device("cpu"))

# with open("0716" + ".txt", "w") as file:
#     for key, value in checkpoint.items():
#         file.write(f"Key: {key}\n")
#         file.write(f"Type: {type(value)}\n")
#         if torch.is_tensor(value):
#             file.write(f"Size: {value.size()}\n")
#             # file.write(f"Value: {value}")
#         file.write("\n")


# with open("old.txt", "w") as file:
#     for key, value in checkpoint.items():
#         file.write(f"Key: {key}\n")
#         file.write(f"Type: {type(value)}\n")
#         if torch.is_tensor(value):
#             file.write(f"Size: {value.size()}\n")
#         file.write("\n")

# for subkey, subvalue in checkpoint["model"].items():
#     file.write(f"Subkey: {subkey}\n")
#     file.write(f"Type: {type(subvalue)}\n")
#     if torch.is_tensor(subvalue):
#         file.write(f"Size: {subvalue.size()}\n")
#     # file.write(f"{subvalue}\n")
#     file.write("\n")

# python /home/ubuntu/louis_crq/AutoFormer_MCU_louis_copy/structure/write_structure.py
