import os
os.chdir('./Segmentation-Digital-Rock-Images')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.residual_unet import ResidualUNet
from utils.dataset import SegmentationDataset
from utils.metrics import iou_per_class, dice_coefficient, calculate_ssim, calculate_psnr
from utils.visualization import show_predictions, plot_metrics
from utils.save import save_checkpoint

from tqdm import tqdm
import matplotlib.pyplot as plt

# Store metrics for plotting
train_metrics = {'loss': [], 'iou': [], 'iou_bg': [], 'iou_fg': [], 'dice': [], 'ssim': [], 'psnr': []}
eval_metrics = {'loss': [], 'iou': [], 'iou_bg': [], 'iou_fg': [], 'dice': [], 'ssim': [], 'psnr': []}

# Loss function: Binary Cross-Entropy with logits
criterion = nn.BCEWithLogitsLoss()

def train_model(model, train_loader, eval_loader, optimizer, device, num_epochs=25):
    model = model.to(device)
    best_eval_loss = float('inf')  # Initialize best evaluation loss

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        iou_score = 0.0
        iou_bg_score = 0.0
        iou_fg_score = 0.0
        dice_score, ssim_score, psnr_score = 0.0, 0.0, 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Apply sigmoid to the outputs and compute metrics
            outputs_sigmoid = torch.sigmoid(outputs)
            iou_bg, iou_fg = iou_per_class(outputs_sigmoid, masks)
            iou_score += (iou_bg + iou_fg) / 2
            iou_bg_score += iou_bg
            iou_fg_score += iou_fg
            dice_score += dice_coefficient(outputs_sigmoid, masks).item()
            ssim_score += calculate_ssim(outputs_sigmoid, masks)
            psnr_score += calculate_psnr(outputs_sigmoid, masks)

        epoch_loss = running_loss / len(train_loader)
        epoch_iou = iou_score / len(train_loader)
        epoch_iou_bg = iou_bg_score / len(train_loader)
        epoch_iou_fg = iou_fg_score / len(train_loader)
        epoch_dice = dice_score / len(train_loader)
        epoch_ssim = ssim_score / len(train_loader)
        epoch_psnr = psnr_score / len(train_loader)

        # Save metrics
        train_metrics['loss'].append(epoch_loss)
        train_metrics['iou'].append(epoch_iou)
        train_metrics['iou_bg'].append(epoch_iou_bg)
        train_metrics['iou_fg'].append(epoch_iou_fg)
        train_metrics['dice'].append(epoch_dice)
        train_metrics['ssim'].append(epoch_ssim)
        train_metrics['psnr'].append(epoch_psnr)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f} '
              f'(BG: {epoch_iou_bg:.4f}, FG: {epoch_iou_fg:.4f}), Dice: {epoch_dice:.4f}, '
              f'SSIM: {epoch_ssim:.4f}, PSNR: {epoch_psnr:.4f}')

        # Evaluate the model after every epoch
        eval_loss = evaluate_model(model, eval_loader, device, epoch+1)

        # Check if this is the best model
        is_best = eval_loss < best_eval_loss
        if is_best:
            best_eval_loss = eval_loss
            print(f"New best model found at epoch {epoch+1} with evaluation loss: {best_eval_loss:.4f}")

        # Save the current model state
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': eval_loss
        }, is_best)

    # After training is done
    plot_metrics(train_metrics, eval_metrics)

def evaluate_model(model, dataloader, device, epoch):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    iou_score = 0.0
    iou_bg_score = 0.0
    iou_fg_score = 0.0
    dice_score, ssim_score, psnr_score = 0.0, 0.0, 0.0

    with torch.no_grad():  # No gradient calculation
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch} - Evaluation"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # Apply sigmoid to the outputs and compute metrics
            outputs_sigmoid = torch.sigmoid(outputs)
            iou_bg, iou_fg = iou_per_class(outputs_sigmoid, masks)
            iou_score += (iou_bg + iou_fg) / 2
            iou_bg_score += iou_bg
            iou_fg_score += iou_fg
            dice_score += dice_coefficient(outputs_sigmoid, masks).item()
            ssim_score += calculate_ssim(outputs_sigmoid, masks)
            psnr_score += calculate_psnr(outputs_sigmoid, masks)

    # Calculate average metrics
    avg_loss = running_loss / len(dataloader)
    avg_iou = iou_score / len(dataloader)
    avg_iou_bg = iou_bg_score / len(dataloader)
    avg_iou_fg = iou_fg_score / len(dataloader)
    avg_dice = dice_score / len(dataloader)
    avg_ssim = ssim_score / len(dataloader)
    avg_psnr = psnr_score / len(dataloader)

    # Save metrics
    eval_metrics['loss'].append(avg_loss)
    eval_metrics['iou'].append(avg_iou)
    eval_metrics['iou_bg'].append(avg_iou_bg)
    eval_metrics['iou_fg'].append(avg_iou_fg)
    eval_metrics['dice'].append(avg_dice)
    eval_metrics['ssim'].append(avg_ssim)
    eval_metrics['psnr'].append(avg_psnr)

    print(f'Evaluation Metrics - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f} '
          f'(BG: {avg_iou_bg:.4f}, FG: {avg_iou_fg:.4f}), Dice: {avg_dice:.4f}, '
          f'SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}')

    # Optionally, visualize predictions after certain epochs
    if epoch % 1 == 0:  # Adjust frequency as needed
        from utils.visualization import show_predictions
        show_predictions(model, dataloader, epoch, device, num_examples=5)

    return avg_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Folder paths
    train_image_folder = './Dataset/Data/images/Training Images'
    train_mask_folder = './Dataset/Data/segmented_masks/Training Images'
    eval_image_folder = './Dataset/Data/images/Validation Images'
    eval_mask_folder = './Dataset/Data/segmented_masks/Validation Images'

    # Load the datasets
    train_dataset = SegmentationDataset(train_image_folder, train_mask_folder)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    eval_dataset = SegmentationDataset(eval_image_folder, eval_mask_folder)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

    # Instantiate and test the model
    model = ResidualUNet()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    # Train the model
    train_model(model, train_loader, eval_loader, optimizer, device, num_epochs=20)

if __name__ == "__main__":
    main()
