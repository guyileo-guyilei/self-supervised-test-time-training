from utils.data_augment import data_augment
from utils.RandomRotation90deg import *
from utils.transforms import *
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Testing Parameters
dtype = torch.float32
print_every = 100
augment_size = 10
n_class = 10

def test_batch(x, y, model, two_head=False, ttt=False, optimizer=None, transform=None,i=-1):
    """
    Test a single batch and return the scores from the model. If needed, also run Test-Time Training on the model.
    
    Inputs:
    - x: A single batch of testing/validation data.
    - y: The corresponding labels to the batch.
    - model: A PyTorch Module giving the model to train.
    - two_head (Optional): A flag signaling if the model is a two-headed model or not.
    - ttt (Optional): A flag signaling if the model will be doing Test-Time Training.
    - optimizer (Optional): An Optimizer object we will use to train the model at test-time.
    - transform (Optional): An object that adds noise to images.
    - i (Optional): Used for displaying sample images at different noise levels.
    
    Returns: The scores of the model given input x
    """
    device = next(model.parameters()).device  # Find device to work with
    
    # Transform test images using a given transform, simulating effects from noise
    if transform is not None:
        x = transform(x)
            
    if ttt:  # ONLY do TTT on test set or TTT val set
        aux_task = RandomRotation90deg()
    
        # Data augmentation on the batch x
        batch_size,_,_,_ = x.shape
        aug_x, aug_y = data_augment(x,y,augment_size=augment_size,include_orig=False)
        aug_x = aug_x.to(device=device, dtype=dtype)  # Make sure augmented data is on device
        aug_y = aug_y.to(device=device, dtype=torch.long)

        # For each augmented batch, do TTT and update SSHead/SharedBranch
        for i in range(augment_size):
            batch = aug_x[i*batch_size:(i+1)*batch_size]
            output,labels = aux_task(batch)  # Auxilliary task data w/ corresponding rotation labels on augmented data

            scores_ss = model(output)[1]  # Only use ss head's result
            loss_ss = F.cross_entropy(scores_ss, labels)
            optimizer.zero_grad()
            loss_ss.backward()
            optimizer.step()

    with torch.no_grad():
        if two_head:
            scores = model(x)[0]  # Only use main head's results for prediction
        else:
            scores = model(x)
    
    # Print sample images
    if i == 0:
        img_clean = transforms.functional.to_pil_image(x[0].squeeze())
        plt.imshow(img_clean,cmap='gray')
        plt.title(f"Clean")
        plt.show()
    elif i == 30:
        img_corrupted = transforms.functional.to_pil_image(x[0].squeeze())
        plt.imshow(img_corrupted,cmap='gray')
        plt.title(f"Low Intensity Corruption")
        plt.show()
    elif i == 150:
        img_corrupted = transforms.functional.to_pil_image(x[0].squeeze())
        plt.imshow(img_corrupted,cmap='gray')
        plt.title(f"High Intensity Corruption")
        plt.show()
    return scores

def check_accuracy(loader, model, two_head=False, ttt=False, optimizer=None, transform=None, printing=True):
    """
    Test the model on a whole Dataloader object. If needed, also run Test-Time Training on the model.
    
    Inputs:
    - loader: The Dataloader object to test on.
    - model: A PyTorch Module to test on.
    - two_head (Optional): A flag signaling if the model is a two-headed model or not.
    - ttt (Optional): A flag signaling if the model will be doing Test-Time Training.
    - optimizer (Optional): An Optimizer object we will use to train the model at test-time.
    - transform (Optional): An object that adds noise to images.
    - printing (Optional): Whether to include the print statements.
    
    Returns: The scores of the model given input x
    """
    device = next(model.parameters()).device  # Find device to work with
    
    if printing:
        if loader.dataset.train:
            print('Checking accuracy on validation set')
        else:
            print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode

    for x, y in loader:
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=torch.long)

        scores = test_batch(x, y, model, two_head=two_head, ttt=ttt, optimizer=optimizer, transform=transform)

        _, preds = scores.max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
        
    acc = float(num_correct) / num_samples
    if printing:
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


def test_increasing_noise(loader, model, two_head=False, ttt=False, optimizer=None, transformClass=None, max_noise=1, sample=False): 
    """
    Test a model, increasing the noise with each batch. If needed, also run Test-Time Training on the model.
    
    Inputs:
    - loader: The Dataloader object to test on.
    - model: A PyTorch Module to test on.
    - two_head (Optional): A flag signaling if the model is a two-headed model or not.
    - ttt (Optional): A flag signaling if the model will be doing Test-Time Training.
    - optimizer (Optional): An Optimizer object we will use to train the model at test-time.
    - transformClass (Optional): A class that adds noise to images. Specified in utils/transforms.py.
    - max_noise (Optional): The maximum noise values reached on the final batch. For Gaussian, this is the StD.
    
    Returns: The scores of the model given input x
    """
    device = next(model.parameters()).device  # Find device to work with
    
    num_incorrect = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    error = []
    
    i = 0
    for x,y in loader:
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=torch.long)

        # Define transforms 
        cur_noise = i * max_noise / len(loader) # Linearly increase noise per batch
        if transformClass is GaussianNoise:
            transform = transformClass(std=cur_noise)
        elif transformClass is not None:
            transform = transformClass(cur_noise)

        else:
            transform = None
        
        if sample:
            cur_batch = i
        else:
            cur_batch = -1
        scores = test_batch(x,y,model,two_head=two_head,ttt=ttt,optimizer=optimizer,transform=transform,i=cur_batch)

        _, preds = scores.max(1)
        
        # Update current error across all batches, appending that value each time a batch finishes
        num_incorrect += (preds != y).sum()
        num_samples += preds.size(0)
        i += 1
        err = float(num_incorrect) / num_samples
        error.append(err)
    return error