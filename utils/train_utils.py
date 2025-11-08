import torch
import torch.nn as nn
from tqdm import tqdm
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference


# --- LOSS DEFINITIONS ---
bce_loss = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss(sigmoid=True)

def combined_loss(pred, target):
    """Hybrid BCE + Dice loss for segmentation."""
    return 0.5 * bce_loss(pred, target) + 0.5 * dice_loss(pred, target)


# --- TRAINING LOOP ---
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"  [Train] Avg Loss: {avg_loss:.4f}")
    return avg_loss


# --- VALIDATION / EVALUATION ---
def evaluate(model, loader, criterion, device, threshold=0.5):
    """Run evaluation on validation or test set."""
    model.eval()
    val_loss = 0.0
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs) >= threshold).float()
            dice_metric(y_pred=preds, y=masks)

    mean_dice = dice_metric.aggregate().item()
    avg_loss = val_loss / len(loader)
    dice_metric.reset()

    print(f"  [Val] Avg Loss: {avg_loss:.4f}, Dice: {mean_dice:.4f}")
    return avg_loss, mean_dice


# --- SLIDING WINDOW EVALUATION ---
def evaluate_model(
    model, dataloader, device, loss_fn,
    threshold=0.5, sw_batch_size=1, roi_size=(128, 160, 160)
):
    """Full 3D sliding-window evaluation for volumetric inference."""
    model.eval()
    running_loss = 0.0
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    n_batches = 0

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = sliding_window_inference(
                imgs, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model
            )

            loss = loss_fn(outputs, masks)
            running_loss += loss.item()

            preds = (torch.sigmoid(outputs) >= threshold).float()
            dice_metric(y_pred=preds, y=masks)
            n_batches += 1

    avg_loss = running_loss / n_batches if n_batches > 0 else float("nan")
    avg_dice = dice_metric.aggregate().item() if n_batches > 0 else float("nan")
    dice_metric.reset()

    print(f"  [Eval] Avg Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")
    return avg_loss, avg_dice
