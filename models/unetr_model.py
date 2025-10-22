import torch
from monai.networks.nets import UNETR

def get_unetr(device):
    return UNETR(
        in_channels=3,
        out_channels=1,
        img_size=(128, 160, 160),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0
    ).to(device)
