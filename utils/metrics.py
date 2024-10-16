# utils/metrics.py

import torch
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def iou_per_class(pred, mask, num_classes=2):
    pred = (pred > 0.5).float()
    mask = mask.float()
    iou_scores = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        mask_cls = (mask == cls).float()
        intersection = (pred_cls * mask_cls).sum()
        union = pred_cls.sum() + mask_cls.sum() - intersection
        if union == 0:
            iou = torch.tensor(1.0)
        else:
            iou = intersection / union
        iou_scores.append(iou.item())
    return iou_scores  # [iou_class0, iou_class1]

def dice_coefficient(pred, mask):
    pred = (pred > 0.5).float()
    intersection = (pred * mask).sum()
    return (2. * intersection) / (pred.sum() + mask.sum() + 1e-6)

def calculate_ssim(pred, mask):
    pred = pred.squeeze().detach().cpu().numpy()
    mask = mask.squeeze().detach().cpu().numpy()
    return ssim(pred, mask, win_size=3, data_range=pred.max() - pred.min())

def calculate_psnr(pred, mask):
    pred = pred.squeeze().detach().cpu().numpy()
    mask = mask.squeeze().detach().cpu().numpy()
    return psnr(mask, pred)
