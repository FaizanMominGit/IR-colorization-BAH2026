import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.dataset import TIRDataset
from src.models.colorization import ThermalColorizerNet
from utils.logging_utils import setup_logging

def train_colorization_model(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = setup_logging(args.checkpoint_dir)
    logger.info("Starting Semantic Colorization Model Training...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using compute device: {device}")

    # Initialize Dataset and Dataloader
    full_dataset = TIRDataset(args.patches_dir, normalize=True)
    if len(full_dataset) == 0:
        logger.error(f"No valid patches found in directory: {args.patches_dir}")
        return

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info(f"Dataset split: {train_size} train samples, {val_size} validation samples.")

    # Initialize Model, Optimizer, Loss Functions
    model = ThermalColorizerNet(in_channels=1, out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            tir_100 = batch['tir_100m'].to(device)
            rgb_100 = batch['rgb_100m'].to(device)

            optimizer.zero_grad()
            color_output = model(tir_100)
            
            if color_output.shape != rgb_100.shape:
                color_output = nn.functional.interpolate(color_output, size=rgb_100.shape[2:], mode='bilinear', align_corners=True)

            loss_l1 = criterion_l1(color_output, rgb_100)
            loss_mse = criterion_mse(color_output, rgb_100)
            loss = loss_l1 + 0.5 * loss_mse

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * tir_100.size(0)

        epoch_train_loss = running_loss / train_size

        # Validation Loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                tir_100 = batch['tir_100m'].to(device)
                rgb_100 = batch['rgb_100m'].to(device)
                color_output = model(tir_100)
                if color_output.shape != rgb_100.shape:
                    color_output = nn.functional.interpolate(color_output, size=rgb_100.shape[2:], mode='bilinear', align_corners=True)
                loss = criterion_l1(color_output, rgb_100) + 0.5 * criterion_mse(color_output, rgb_100)
                running_val_loss += loss.item() * tir_100.size(0)

        epoch_val_loss = running_val_loss / max(val_size, 1)

        logger.info(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, 'colorizer_model_best.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved best colorizer model checkpoint to {checkpoint_path}")

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'colorizer_model_final.pth'))
    logger.info("Semantic Colorization Training Completed Successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Semantic Colorization Model for TIR Imagery.")
    parser.add_argument('--patches_dir', type=str, default='output/patches', help='Path to dataset patches.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split fraction.')

    args = parser.parse_args()
    train_colorization_model(args)
