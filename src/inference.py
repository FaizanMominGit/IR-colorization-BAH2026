import os
import glob
import time
import argparse
import logging
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from src.models import ThermalSRNet, ThermalColorizerNet
from utils.logging_utils import setup_logging
from utils.file_utils import find_file, read_tif_image, write_tif_image

def run_inference(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output', 'model_outputs')
    sr_out_dir = os.path.join(output_dir, 'tir_superresolved_100m')
    col_out_dir = os.path.join(output_dir, 'colorized_tir_100m')

    os.makedirs(sr_out_dir, exist_ok=True)
    os.makedirs(col_out_dir, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("Starting Pipeline Inference Execution...")

    # Load Models if PyTorch is available
    sr_model, color_model, device = None, None, None
    if HAS_TORCH and ThermalSRNet is not None and ThermalColorizerNet is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using PyTorch compute device: {device}")

        sr_model = ThermalSRNet(in_channels=1, out_channels=1, scale_factor=2).to(device)
        color_model = ThermalColorizerNet(in_channels=1, out_channels=3).to(device)

        if os.path.exists(args.sr_weights):
            sr_model.load_state_dict(torch.load(args.sr_weights, map_location=device))
            logger.info(f"Loaded SR model weights from {args.sr_weights}")
        else:
            logger.warning(f"SR weights not found at {args.sr_weights}. Operating in baseline initialization mode.")

        if os.path.exists(args.color_weights):
            color_model.load_state_dict(torch.load(args.color_weights, map_location=device))
            logger.info(f"Loaded Colorizer model weights from {args.color_weights}")
        else:
            logger.warning(f"Colorizer weights not found at {args.color_weights}. Operating in baseline initialization mode.")

        sr_model.eval()
        color_model.eval()
    else:
        logger.info("Operating in framework-agnostic mode using baseline image interpolation heuristics.")

    # Find candidate inputs
    input_root = os.path.join(base_dir, 'input')
    downscaled_dir = os.path.join(base_dir, 'output', 'downscaled_data')

    product_ids = []
    if os.path.exists(input_root):
        product_ids = [e for e in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, e))]

    if not product_ids:
        logger.warning("No product IDs found in input directory to run inference on.")
        return

    for product_id in product_ids:
        start_time = time.time()
        logger.info(f"Processing inference for product: {product_id}")

        tir_200_path = find_file(downscaled_dir, f'{product_id}*_tir_200m*')
        if not tir_200_path:
            input_dir = os.path.join(input_root, product_id)
            tir_200_path = find_file(input_dir, '_B10')

        if not tir_200_path:
            logger.warning(f"Skipping {product_id}: No TIR band file found.")
            continue

        try:
            raw_tir = read_tif_image(tir_200_path).astype(np.float32)
            if raw_tir.ndim == 3:
                raw_tir = raw_tir[0]

            h_orig, w_orig = raw_tir.shape
            p_low, p_high = np.percentile(raw_tir, (1, 99))
            norm_tir = np.clip((raw_tir - p_low) / (p_high - p_low + 1e-6), 0.0, 1.0)

            if sr_model is not None and color_model is not None:
                t_in = torch.from_numpy(norm_tir).float().unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    t_sr = sr_model(t_in)
                    t_col = color_model(t_sr)

                sr_arr = t_sr.squeeze().cpu().numpy()
                col_arr = t_col.squeeze().cpu().numpy()
            else:
                # Fallback baseline upsampling & multi-spectral translation mapping
                import cv2
                sr_arr = cv2.resize(norm_tir, (w_orig * 2, h_orig * 2), interpolation=cv2.INTER_CUBIC)
                col_b = sr_arr * 0.8
                col_g = sr_arr * 0.9
                col_r = sr_arr * 1.0
                col_arr = np.stack([col_b, col_g, col_r], axis=0)

            sr_denorm = sr_arr * (p_high - p_low + 1e-6) + p_low
            col_bgr = col_arr.astype(np.float32)

            sr_out_path = os.path.join(sr_out_dir, f'{product_id}.tif')
            col_out_path = os.path.join(col_out_dir, f'{product_id}.tif')

            write_tif_image(sr_out_path, sr_denorm.astype(np.float32))
            write_tif_image(col_out_path, col_bgr.astype(np.float32))

            elapsed = time.time() - start_time
            logger.info(f"Successfully processed scene {product_id} in {elapsed:.3f} seconds.")
            logger.info(f"Saved Super-Resolved GeoTIFF (100m): {sr_out_path}")
            logger.info(f"Saved Colorized GeoTIFF (100m, BGR): {col_out_path}")

        except Exception as e:
            logger.error(f"Error during inference on {product_id}: {e}")

    logger.info("Pipeline Inference Finished Successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run full pipeline inference on satellite products.")
    parser.add_argument('--sr_weights', type=str, default='checkpoints/sr_model_best.pth', help='Path to SR weights.')
    parser.add_argument('--color_weights', type=str, default='checkpoints/colorizer_model_best.pth', help='Path to Colorizer weights.')
    args = parser.parse_args()
    run_inference(args)
