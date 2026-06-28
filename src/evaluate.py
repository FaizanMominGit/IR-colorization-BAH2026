import os
import argparse
import logging
import numpy as np

from utils.logging_utils import setup_logging
from utils.file_utils import find_file, read_tif_image

def numpy_psnr(target, ref, data_range=None):
    """Computes Peak Signal-to-Noise Ratio using pure NumPy."""
    target = target.astype(np.float64)
    ref = ref.astype(np.float64)
    mse = np.mean((target - ref) ** 2)
    if mse == 0:
        return float('inf')
    if data_range is None:
        data_range = ref.max() - ref.min()
        if data_range == 0:
            data_range = 1.0
    return 10.0 * np.log10((data_range ** 2) / mse)

def numpy_ssim(img1, img2, data_range=None):
    """Computes basic Structural Similarity Index (SSIM) using pure NumPy."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if data_range is None:
        data_range = max(img1.max() - img1.min(), 1.0)
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim_num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim_den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return ssim_num / ssim_den

def compute_color_mae(pred_rgb, gt_rgb):
    """
    Computes Mean Absolute Error (MAE) across RGB color channels.
    Assumes channels format (3, H, W).
    """
    if pred_rgb.ndim == 3 and pred_rgb.shape[0] == 3:
        pass
    elif pred_rgb.ndim == 3 and pred_rgb.shape[-1] == 3:
        pred_rgb = np.moveaxis(pred_rgb, -1, 0)

    if gt_rgb.ndim == 3 and gt_rgb.shape[0] == 3:
        pass
    elif gt_rgb.ndim == 3 and gt_rgb.shape[-1] == 3:
        gt_rgb = np.moveaxis(gt_rgb, -1, 0)

    min_h = min(pred_rgb.shape[1], gt_rgb.shape[1])
    min_w = min(pred_rgb.shape[2], gt_rgb.shape[2])

    p_crop = pred_rgb[:, :min_h, :min_w].astype(np.float32)
    g_crop = gt_rgb[:, :min_h, :min_w].astype(np.float32)

    if g_crop.max() > 1.0:
        p_low, p_high = np.percentile(g_crop, (1, 99))
        if p_high > p_low:
            g_crop = np.clip((g_crop - p_low) / (p_high - p_low), 0.0, 1.0)
        else:
            g_crop = np.clip(g_crop / 65535.0, 0.0, 1.0)

    mae_blue = np.mean(np.abs(p_crop[0] - g_crop[0]))
    mae_green = np.mean(np.abs(p_crop[1] - g_crop[1]))
    mae_red = np.mean(np.abs(p_crop[2] - g_crop[2]))
    total_mae = np.mean(np.abs(p_crop - g_crop))

    return total_mae, (mae_blue, mae_green, mae_red)

def evaluate_outputs(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output')
    logger = setup_logging(output_dir)
    logger.info("Starting Framework-Agnostic Evaluation Metrics Calculation...")

    model_outputs_dir = os.path.join(output_dir, 'model_outputs')
    sr_pred_dir = os.path.join(model_outputs_dir, 'tir_superresolved_100m')
    col_pred_dir = os.path.join(model_outputs_dir, 'colorized_tir_100m')
    gt_dir = os.path.join(output_dir, 'downscaled_data')

    if not os.path.exists(sr_pred_dir) or not os.path.exists(col_pred_dir):
        logger.warning(f"Model output directory {model_outputs_dir} not found. Ready for post-inference evaluation.")
        return

    sr_files = [f for f in os.listdir(sr_pred_dir) if f.endswith('.tif')]

    if not sr_files:
        logger.warning("No generated output TIFFs found in output/model_outputs to evaluate.")
        return

    psnr_list, ssim_list, color_mae_list = [], [], []

    for f_name in sr_files:
        product_id = os.path.splitext(f_name)[0]
        logger.info(f"Evaluating product scene: {product_id}")

        sr_pred_path = os.path.join(sr_pred_dir, f_name)
        col_pred_path = os.path.join(col_pred_dir, f_name)

        gt_tir_path = find_file(gt_dir, f'{product_id}*_tir_100m*')
        gt_rgb_path = find_file(gt_dir, f'{product_id}*_rgb_100m*')

        if not gt_tir_path or not gt_rgb_path:
            logger.warning(f"Ground truth files for {product_id} missing in {gt_dir}. Skipping metric computation.")
            continue

        pred_sr = np.squeeze(read_tif_image(sr_pred_path))
        pred_col = read_tif_image(col_pred_path)
        gt_tir = np.squeeze(read_tif_image(gt_tir_path))
        gt_rgb = read_tif_image(gt_rgb_path)

        min_h = min(pred_sr.shape[-2], gt_tir.shape[-2])
        min_w = min(pred_sr.shape[-1], gt_tir.shape[-1])

        p_sr_crop = pred_sr[..., :min_h, :min_w]
        gt_tir_crop = gt_tir[..., :min_h, :min_w]


        data_range = float(gt_tir_crop.max() - gt_tir_crop.min() + 1e-6)
        psnr_val = numpy_psnr(p_sr_crop, gt_tir_crop, data_range=data_range)
        ssim_val = numpy_ssim(p_sr_crop, gt_tir_crop, data_range=data_range)

        c_mae, (b_mae, g_mae, r_mae) = compute_color_mae(pred_col, gt_rgb)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        color_mae_list.append(c_mae)

        logger.info(f"[{product_id}] SR Quality   -> PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
        logger.info(f"[{product_id}] Color Quality -> Total MAE: {c_mae:.4f} (Blue: {b_mae:.4f}, Green: {g_mae:.4f}, Red: {r_mae:.4f})")

    if psnr_list:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_c_mae = np.mean(color_mae_list)
        logger.info("=" * 60)
        logger.info(f"FINAL EVALUATION METRICS across {len(psnr_list)} products:")
        logger.info(f"  - Average Super-Resolution PSNR: {avg_psnr:.2f} dB")
        logger.info(f"  - Average Super-Resolution SSIM: {avg_ssim:.4f}")
        logger.info(f"  - Average Colorization MAE    : {avg_c_mae:.4f}")
        logger.info("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate pipeline model outputs against ground truth.")
    args = parser.parse_args()
    evaluate_outputs(args)
