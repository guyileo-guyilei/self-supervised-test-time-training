import kornia.augmentation as K
import torch.nn as nn
import torch

def data_augment(batch,labels,augment_size=3,include_orig=True):
    transform = nn.Sequential(
                    K.RandomResizedCrop(size=(28, 28),scale=(.8,1)),
                    K.RandomHorizontalFlip(p=.5),
                )
    
    batch_repeated = batch.repeat_interleave(augment_size, dim=0)
    augmented_batch = transform(batch_repeated)
    augmented_labels = labels.repeat_interleave(augment_size)
    
    if include_orig:
        augmented_batch = torch.cat([batch, augmented_batch], dim=0)
        try:
            augmented_labels = torch.cat([labels, augmented_labels], dim=0)
        except RuntimeError:  # Zero dimensional tensor (single value for TTT)
            augmented_labels = torch.cat([labels.view(1), augmented_labels], dim=0)
        
    return augmented_batch, augmented_labels