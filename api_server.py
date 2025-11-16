"""
API Server for Brain Tumor Segmentation
Provides REST endpoints for the frontend to interact with trained models
Uses existing project modules - no code duplication!
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import os
import io
import base64
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import YOUR existing modules - no duplication!
from config import DEVICE as CONFIG_DEVICE, BASE_DIR
from utils.predict_eval_utils import predict, evaluate
from models.unetr_model import get_unetr
from datasets.brain_tumor_dataset import BrainTumor3DDataset
from monai.transforms import DivisiblePad

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'shared/global'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Device configuration
device = torch.device(CONFIG_DEVICE if torch.cuda.is_available() and CONFIG_DEVICE == 'cuda' else 'cpu')
print(f"API Server running on device: {device}")

# Cache for loaded models
loaded_models = {}


def load_model(model_path):
    """Load a trained model from checkpoint using YOUR get_unetr function"""
    if model_path in loaded_models:
        return loaded_models[model_path]
    
    try:
        # Use YOUR existing get_unetr function!
        model = get_unetr(device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        loaded_models[model_path] = model
        print(f"✓ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def preprocess_single_sample(img_array, mask_array):
    """
    Apply the EXACT same preprocessing as BrainTumor3DDataset.__getitem__
    This ensures consistency between training and inference!
    """
    print(f"Raw input shapes - Image: {img_array.shape}, Mask: {mask_array.shape}")
    
    # Step 1: Reorder dimensions (same as your dataset)
    # Image: [H, W, D, C] -> [C, D, H, W]
    img = np.transpose(img_array, (3, 2, 0, 1))
    
    # Mask: [H, W, D] -> [D, H, W]
    mask = np.transpose(mask_array, (2, 0, 1))
    
    # Step 2: Add channel dimension to mask
    mask = np.expand_dims(mask, axis=0)  # [1, D, H, W]
    
    print(f"After transpose - Image: {img.shape}, Mask: {mask.shape}")
    
    # Step 3: Convert to PyTorch tensors
    img = torch.from_numpy(img).float()
    mask = torch.from_numpy(mask).float()
    
    # Step 4: Apply padding (divisible by 16, same as your dataset)
    pad = DivisiblePad(k=16)
    img = pad(img)
    mask = pad(mask)
    
    print(f"After padding - Image: {img.shape}, Mask: {mask.shape}")
    
    return img, mask


def create_visualization(image, mask, pred_mask, slice_idx):
    """Create side-by-side comparison visualization for specified slice"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image (first channel, specified slice)
    axes[0].imshow(image[0, slice_idx, :, :], cmap='gray')
    axes[0].set_title(f'Original MRI Scan (Slice {slice_idx})', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(mask[0, slice_idx, :, :], cmap='jet', alpha=0.8, vmin=0, vmax=1)
    axes[1].set_title('Ground Truth (Doctor)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Predicted mask
    axes[2].imshow(pred_mask[0, slice_idx, :, :], cmap='jet', alpha=0.8, vmin=0, vmax=1)
    axes[2].set_title('Model Prediction', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Convert to base64 for web transfer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'base_dir': BASE_DIR
    })


@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available trained models"""
    models = []
    models_path = Path(MODELS_FOLDER)
    
    if not models_path.exists():
        return jsonify({
            'models': [],
            'error': f'Models folder not found: {MODELS_FOLDER}. Run training first!'
        }), 404
    
    # Find all .pth files
    pth_files = sorted(models_path.glob('*.pth'))
    
    if not pth_files:
        return jsonify({
            'models': [],
            'error': 'No trained models found. Run train_federated.py first!'
        }), 404
    
    # Add latest model first (if exists)
    latest_path = models_path / 'global_latest.pth'
    if latest_path.exists():
        models.append({
            'name': 'global_latest',
            'display_name': 'Global Model (Latest)',
            'path': str(latest_path),
            'size_mb': round(latest_path.stat().st_size / (1024 * 1024), 2)
        })
    
    # Add round-specific models
    for idx, model_file in enumerate(pth_files):
        if model_file.name != 'global_latest.pth':
            round_num = model_file.stem.replace('global_round_', '')
            models.append({
                'name': model_file.stem,
                'display_name': f'Global Model (Round {round_num})',
                'path': str(model_file),
                'size_mb': round(model_file.stat().st_size / (1024 * 1024), 2)
            })
    
    return jsonify({'models': models})


@app.route('/api/segment', methods=['POST'])
def segment_image():
    """
    Perform segmentation on uploaded image and mask
    Uses YOUR existing predict() and evaluate() functions
    """
    try:
        # Validate file uploads
        if 'image' not in request.files or 'mask' not in request.files:
            return jsonify({'error': 'Both image and mask files required'}), 400
        
        image_file = request.files['image']
        mask_file = request.files['mask']
        
        if image_file.filename == '' or mask_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Get model selection and slice index
        model_name = request.form.get('model', 'global_latest')
        slice_idx = int(request.form.get('slice_idx', -1))  # -1 means middle slice (default)
        
        model_path = os.path.join(MODELS_FOLDER, f"{model_name}.pth")
        
        if not os.path.exists(model_path):
            return jsonify({
                'error': f'Model not found: {model_path}',
                'hint': 'Run train_federated.py to create the model'
            }), 404
        
        print(f"\n{'='*70}")
        print(f"Processing Segmentation Request")
        print(f"{'='*70}")
        print(f"Image: {image_file.filename}")
        print(f"Mask:  {mask_file.filename}")
        print(f"Model: {model_name}")
        print(f"{'='*70}\n")
        
        # Save files temporarily
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        mask_path = os.path.join(UPLOAD_FOLDER, mask_file.filename)
        image_file.save(image_path)
        mask_file.save(mask_path)
        
        # Load numpy arrays
        img_array = np.load(image_path)
        mask_array = np.load(mask_path)
        
        # Preprocess using YOUR dataset's exact logic
        image_tensor, mask_tensor = preprocess_single_sample(img_array, mask_array)
        
        # Get depth dimension for slice validation
        depth = image_tensor.shape[1]  # Shape is [C, D, H, W]
        
        # Validate or set slice index
        if slice_idx < 0 or slice_idx >= depth:
            slice_idx = depth // 2  # Use middle slice as default
        
        print(f"Using slice {slice_idx} out of {depth} total slices")
        
        # Move to device
        image_tensor = image_tensor.to(device)
        mask_tensor = mask_tensor.to(device)
        
        # Load model using YOUR get_unetr function
        model = load_model(model_path)
        
        # Perform prediction using YOUR predict function
        print("Running prediction with sliding window inference...")
        pred_mask = predict(
            model=model,
            image=image_tensor,
            device=device,
            threshold=0.5,
            roi_size=(128, 160, 160),
            sw_batch_size=1
        )
        
        # Calculate Dice score only
        print("Calculating Dice score...")
        dice_score, _ = evaluate(pred_mask, mask_tensor.cpu())
        
        print(f"\n{'='*70}")
        print(f"RESULTS:")
        print(f"  Dice Score:      {dice_score:.4f} ({dice_score*100:.2f}%)")
        print(f"  Slice Displayed: {slice_idx}/{depth-1}")
        print(f"{'='*70}\n")
        
        # Create visualization for specified slice
        image_np = image_tensor.cpu().numpy()
        mask_np = mask_tensor.cpu().numpy()
        pred_mask_np = pred_mask.cpu().numpy()
        
        visualization = create_visualization(image_np, mask_np, pred_mask_np, slice_idx)
        
        # Clean up temporary files
        os.remove(image_path)
        os.remove(mask_path)
        
        return jsonify({
            'success': True,
            'model_used': model_name,
            'metrics': {
                'dice_score': float(dice_score),
                'dice_percentage': round(float(dice_score) * 100, 2)
            },
            'visualization': visualization,
            'info': {
                'image_file': image_file.filename,
                'mask_file': mask_file.filename,
                'input_shape': str(img_array.shape),
                'processed_shape': str(tuple(image_tensor.shape)),
                'slice_idx': slice_idx,
                'total_slices': depth
            }
        })
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        trace = traceback.format_exc()
        
        print(f"\n{'='*70}")
        print(f"ERROR:")
        print(trace)
        print(f"{'='*70}\n")
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': trace
        }), 500


@app.route('/api/test-dataset', methods=['GET'])
def test_dataset():
    """Test endpoint to verify YOUR dataset loading works"""
    try:
        from datasets.brain_tumor_dataset import get_client_data
        
        print("Testing dataset loading for client 1...")
        _, _, test_loader = get_client_data(client_id=1, batch_size=1)
        
        # Get one sample
        image, mask = next(iter(test_loader))
        
        print(f"✓ Dataset test successful!")
        print(f"  Sample image shape: {image.shape}")
        print(f"  Sample mask shape:  {mask.shape}")
        print(f"  Total test samples: {len(test_loader.dataset)}")
        
        return jsonify({
            'success': True,
            'message': 'Dataset loading works correctly!',
            'sample_shapes': {
                'image': list(image.shape),
                'mask': list(mask.shape)
            },
            'num_test_samples': len(test_loader.dataset)
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print(" " * 15 + "🧠 Brain Tumor Segmentation API Server 🧠")
    print("=" * 70)
    print(f"Device:          {device}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    print(f"Base Directory:  {BASE_DIR}")
    print(f"Models Folder:   {MODELS_FOLDER}")
    print("-" * 70)
    print("📡 Endpoints:")
    print("  GET  /api/health          - Server health check")
    print("  GET  /api/models          - List available trained models")
    print("  POST /api/segment         - Segment brain MRI scan")
    print("  GET  /api/test-dataset    - Test dataset loading")
    print("-" * 70)
    print("🚀 Server starting on http://localhost:5000")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)