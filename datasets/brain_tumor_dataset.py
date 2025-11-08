import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import DivisiblePad
from glob import glob
from config import BASE_DIR


class BrainTumor3DDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_files = sorted(glob(os.path.join(img_dir, "*.npy")))
        self.mask_files = sorted(glob(os.path.join(mask_dir, "*.npy")))
        self.pad = DivisiblePad(k=16)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = np.load(self.img_files[idx])
        mask = np.load(self.mask_files[idx])

        # Reorder dimensions (assuming shape [H, W, D, C])
        img = np.transpose(img, (3, 2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        img = self.pad(img)
        mask = self.pad(mask)

        return img, mask


def get_client_data(client_id, batch_size=1):

    data_path=os.path.join(BASE_DIR,"data")
    client_path=os.path.join(data_path,f"client_{client_id}")

    train_img = os.path.join(client_path,"Training","images")
    train_mask = os.path.join(client_path,"Training","masks")

    val_img = os.path.join(client_path,"Validation","images")
    val_mask = os.path.join(client_path,"Validation","masks")

    test_img = os.path.join(client_path,"Testing","images")
    test_mask = os.path.join(client_path,"Testing","masks")

    train_ds = BrainTumor3DDataset(train_img, train_mask)
    val_ds = BrainTumor3DDataset(val_img, val_mask)
    test_ds = BrainTumor3DDataset(test_img, test_mask)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
