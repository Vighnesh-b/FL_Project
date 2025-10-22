import os
import numpy as np
import torch
from torch.utils.data import Dataset
from monai.transforms import DivisiblePad
from glob import glob
from torch.utils.data import DataLoader

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
        img = np.transpose(img, (3, 2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        img = self.pad(img)
        mask = self.pad(mask)
        return img, mask
    
def get_client_dataloaders(client_id, batch_size=1):
    #BASE DIR NEEDS TO BE ADDED
    BASE_DIR = ""
    train_img = os.path.join(BASE_DIR, "Training", f"client_{client_id}", "images")
    train_mask = os.path.join(BASE_DIR, "Training", f"client_{client_id}", "masks")

    val_img = os.path.join(BASE_DIR, "Validation", f"client_{client_id}", "images")
    val_mask = os.path.join(BASE_DIR, "Validation", f"client_{client_id}", "masks")

    test_img = os.path.join(BASE_DIR, "Testing", f"client_{client_id}", "images")
    test_mask = os.path.join(BASE_DIR, "Testing", f"client_{client_id}", "masks")

    train_ds = BrainTumor3DDataset(train_img, train_mask)
    val_ds = BrainTumor3DDataset(val_img, val_mask)
    test_ds = BrainTumor3DDataset(test_img, test_mask)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader
