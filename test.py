import numpy as np
import torch
import matplotlib.pyplot as plt
from models.unetr_model import get_unetr
from utils.predict_eval_utils import predict,evaluate,evaluate_per_slice # your predict() function

# ---- Load and prepare image/mask ----
img = np.load("data/client_1/Testing/images/image_223.npy")     # shape (150, 150, 128, 3)
mask = np.load("data/client_1/Testing/masks/mask_223.npy")     # shape (150, 150, 128)

# Reorder to channel-first for the model
img_t = torch.from_numpy(np.transpose(img, (3, 0, 1, 2))).float()
mask_t = torch.from_numpy(np.expand_dims(mask, axis=0)).float()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_unetr(device)
model.load_state_dict(torch.load("shared/global/global_latest.pth", map_location=device))

# ---- Predict segmentation ----
pred_mask = predict(model, img_t, device=device)  # shape (1, H, W, D)
pred_mask_np = pred_mask.squeeze(0).cpu().numpy()  # (H, W, D)

pred_mask_t = torch.from_numpy(pred_mask_np).float()
mask_t = torch.from_numpy(mask).float().unsqueeze(0)  # add channel dim to match shape
print(evaluate(pred_mask_t, mask_t))

# ---- Visualization ----
# Choose some middle slices (along depth axis)
depth = img.shape[2]
slices_to_show = [depth // 4, depth // 2, 3 * depth // 4]

plt.figure(figsize=(15, 10))
for i, z in enumerate(slices_to_show):
    # Convert image slice (H, W, 3)
    img_slice = img[:, :, z, :]
    mask_slice = mask[:, :, z]
    pred_slice = pred_mask_np[:, :, z]

    # Plot image
    plt.subplot(len(slices_to_show), 3, 3 * i + 1)
    plt.imshow(img_slice.astype(np.float32) / np.max(img_slice))
    plt.title(f"Image (Slice {z})")
    plt.axis("off")

    # Plot ground truth mask
    plt.subplot(len(slices_to_show), 3, 3 * i + 2)
    plt.imshow(mask_slice, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    # Plot predicted mask
    plt.subplot(len(slices_to_show), 3, 3 * i + 3)
    plt.imshow(pred_slice, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

plt.tight_layout()
plt.show()
