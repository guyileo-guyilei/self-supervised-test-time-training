import torch
import torch.nn as nn
import kornia.geometry.transform as KGT  # for rotation transform
import numpy as np

class RandomRotation90deg(nn.Module):
    """
    Custom transform that randomly rotates each image in the batch by
    one of the four multiples of 90° and returns both the rotated images and labels.
    """
    def __init__(self):
        super().__init__()
        self.angles = np.array([0, 90, 180, 270])
    
    def forward(self, batch):
        batch_size,_,_,_ = batch.shape
        rotated_images = []
        rotation_labels = []
        
        angle_idx = torch.randint(0, 4, (batch_size,))
        angles = self.angles[angle_idx]
        angles = (torch.from_numpy(angles)
                  .to(device=batch.device)
                  .float()
                 )
        
        rotated_batch = KGT.rotate(batch,angles)        
        return rotated_batch.to(device=batch.device), angle_idx.to(device=batch.device)