import torch
import os

# Function to save the model
def save_checkpoint(state, is_best, folder='checkpoints'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(state, os.path.join(folder, 'checkpoint.pth'))
    if is_best:
        torch.save(state, os.path.join(folder, 'best_model.pth'))
