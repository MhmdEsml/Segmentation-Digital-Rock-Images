# utils/visualization.py
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def show_predictions(model, dataloader, epoch, device, num_examples=5, save_dir='predictions'):
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    model.eval()  # Set model to evaluation mode
    images_shown = 0
    with torch.no_grad():  # No need to calculate gradients during inference
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid to the outputs

            outputs = (outputs > 0.5).float()  # Binarize the predictions

            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            outputs = outputs.cpu().numpy()

            for i in range(images.shape[0]):
                if images_shown >= num_examples:
                    return

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(images[i].squeeze(), cmap='gray')
                axes[0].set_title('Input Image')
                axes[1].imshow(masks[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[1].set_title('Ground Truth Mask')
                axes[2].imshow(outputs[i].squeeze(), cmap='gray', vmin=0, vmax=1)
                axes[2].set_title('Predicted Mask')

                plt.savefig(os.path.join(save_dir, f'Prediction_Epoch_{epoch}_Example_{images_shown + 1}.png'))
                plt.close()  # Close the figure to free memory
                images_shown += 1

def plot_metrics(train_metrics, eval_metrics, save_dir='metrics'):
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    epochs = range(1, len(train_metrics['loss']) + 1)

    plt.figure(figsize=(18, 12))

    # Loss
    plt.subplot(3, 3, 1)
    plt.plot(epochs, train_metrics['loss'], label='Train Loss')
    plt.plot(epochs, eval_metrics['loss'], label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    # IoU
    plt.subplot(3, 3, 2)
    plt.plot(epochs, train_metrics['iou'], label='Train IoU')
    plt.plot(epochs, eval_metrics['iou'], label='Eval IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('IoU')
    plt.legend()

    # IoU Background
    plt.subplot(3, 3, 3)
    plt.plot(epochs, train_metrics['iou_bg'], label='Train IoU FG')
    plt.plot(epochs, eval_metrics['iou_bg'], label='Eval IoU FG')
    plt.xlabel('Epoch')
    plt.ylabel('IoU FG')
    plt.title('IoU Foreground')
    plt.legend()

    # IoU Foreground
    plt.subplot(3, 3, 4)
    plt.plot(epochs, train_metrics['iou_fg'], label='Train IoU BG')
    plt.plot(epochs, eval_metrics['iou_fg'], label='Eval IoU BG')
    plt.xlabel('Epoch')
    plt.ylabel('IoU BG')
    plt.title('IoU Background')
    plt.legend()

    # Dice
    plt.subplot(3, 3, 5)
    plt.plot(epochs, train_metrics['dice'], label='Train Dice')
    plt.plot(epochs, eval_metrics['dice'], label='Eval Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient')
    plt.legend()

    # SSIM
    plt.subplot(3, 3, 6)
    plt.plot(epochs, train_metrics['ssim'], label='Train SSIM')
    plt.plot(epochs, eval_metrics['ssim'], label='Eval SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM')
    plt.legend()

    # PSNR
    plt.subplot(3, 3, 7)
    plt.plot(epochs, train_metrics['psnr'], label='Train PSNR')
    plt.plot(epochs, eval_metrics['psnr'], label='Eval PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('PSNR')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics.png'))
    plt.close()  # Close the figure to free memory
