import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.train_utils import train_one_epoch, evaluate_model, combined_loss
from datasets.brain_tumor_dataset import BrainTumor3DDataset

def run_client(client_id, global_weights, epochs, batch_size, device):
    base_dir = "./data"
    train_img = f"{base_dir}/client_{client_id}/images"
    train_mask = f"{base_dir}/client_{client_id}/masks"
    
    train_ds = BrainTumor3DDataset(train_img, train_mask)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    from models.unetr_model import get_unetr
    model = get_unetr(device)
    if global_weights:
        model.load_state_dict(global_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, combined_loss, device, scaler)
        print(f"[Client {client_id}] Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f}")

    return model.state_dict()
