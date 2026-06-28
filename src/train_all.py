import os
import argparse
import logging
from utils.logging_utils import setup_logging
from src.train_sr import train_sr_model
from src.train_colorization import train_colorization_model

def run_unified_training(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = setup_logging(args.checkpoint_dir)
    logger.info("==========================================================")
    logger.info("STARTING UNIFIED PIPELINE TRAINING ORCHESTRATOR")
    logger.info("==========================================================")

    # 1. Train Super-Resolution Model
    logger.info("--- Phase 1/2: Training Super-Resolution Model ---")
    sr_args = argparse.Namespace(
        patches_dir=args.patches_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs_sr,
        batch_size=args.batch_size,
        lr=args.lr_sr,
        val_split=args.val_split
    )
    try:
        train_sr_model(sr_args)
        logger.info("Phase 1/2 Completed Successfully.")
    except Exception as e:
        logger.error(f"Error during Super-Resolution training phase: {e}")

    # 2. Train Colorization Model
    logger.info("--- Phase 2/2: Training Semantic Colorization Model ---")
    col_args = argparse.Namespace(
        patches_dir=args.patches_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs_color,
        batch_size=args.batch_size,
        lr=args.lr_color,
        val_split=args.val_split
    )
    try:
        train_colorization_model(col_args)
        logger.info("Phase 2/2 Completed Successfully.")
    except Exception as e:
        logger.error(f"Error during Colorization training phase: {e}")

    logger.info("==========================================================")
    logger.info("UNIFIED PIPELINE TRAINING COMPLETED")
    logger.info(f"Checkpoints saved in: {os.path.abspath(args.checkpoint_dir)}")
    logger.info("==========================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified Training Orchestrator for IR Colorization & SR.")
    parser.add_argument('--patches_dir', type=str, default='output/patches', help='Path to dataset patches.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model weights.')
    parser.add_argument('--epochs_sr', type=int, default=5, help='Number of epochs for SR model.')
    parser.add_argument('--epochs_color', type=int, default=5, help='Number of epochs for Colorizer model.')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size.')
    parser.add_argument('--lr_sr', type=float, default=1e-4, help='Learning rate for SR model.')
    parser.add_argument('--lr_color', type=float, default=2e-4, help='Learning rate for Colorizer model.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation set split fraction.')

    args = parser.parse_args()
    run_unified_training(args)
