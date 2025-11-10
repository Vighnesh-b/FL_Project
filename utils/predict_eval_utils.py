import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
import numpy as np

def predict(model, image, device, threshold=0.5, roi_size=(128, 160, 160), sw_batch_size=1):

    model.eval()
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = sliding_window_inference(image, roi_size, sw_batch_size, model)
        pred_mask = (torch.sigmoid(output) >= threshold).float()

    return pred_mask.squeeze(0).cpu()  # Remove batch dimension and move to CPU

def evaluate(pred_mask, true_mask):

    device = pred_mask.device
    pred_mask = pred_mask.unsqueeze(0).unsqueeze(0).float()  # [B, C, H, W, D]
    true_mask = true_mask.unsqueeze(0).unsqueeze(0).float().to(device)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric(y_pred=pred_mask, y=true_mask)
    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()

    correct = (pred_mask == true_mask).float().sum()
    total = true_mask.numel()
    pixel_acc = (correct / total).item()

    return dice_score, pixel_acc

def evaluate_per_slice(pred_mask, mask):
    dice_scores = []
    for i in range(pred_mask.shape[-1]):  # iterate over depth
        pred_slice = pred_mask[..., i]
        mask_slice = mask[..., i]
        intersection = (pred_slice * mask_slice).sum()
        dice = (2. * intersection) / (pred_slice.sum() + mask_slice.sum() + 1e-8)
        dice_scores.append(dice.item())
    return dice_scores




