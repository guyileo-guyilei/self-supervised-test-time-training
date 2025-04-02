from training_testing.testing import check_accuracy
from utils.data_augment import data_augment
from utils.RandomRotation90deg import *
import torch
import torch.nn.functional as F
import torch.optim as optim

# Training parameters
print_every = 100
dtype = torch.float32
augment_size = 10
n_class = 10

def train(model, optimizer, loader_train, loader_val, epochs=1, two_head=False):
    """
    Train a model on Fashion MNIST dataset, using a predefined scheduler and auxilliary task for self-supervised learning.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    - two_head: (Optional) Whether the function expects a two-headed model with self-supervised training. Defaults to False.
    
    Returns: The accuracy of the model
    """
    device = next(model.parameters()).device  # Find device to work with
    aux_task = RandomRotation90deg()  # initialize auxilliary task for SSHead
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.3,patience=1,threshold=1e-3)
    
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            optimizer.zero_grad()
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            # Data augmentation on the single test image x
            aug_x, aug_y = data_augment(x,y,augment_size=augment_size)
            aug_x = aug_x.to(device=device, dtype=dtype)  # Make sure augmented data is on device
            aug_y = aug_y.to(device=device, dtype=torch.long)

            scores = model(aug_x)  # for main head
            if two_head:  # Collect two losses and sum them to optimize together
                # SS training should be based on auxilliary task, main is based on image classification
                aux_x,aux_y = aux_task(aug_x)  # Auxilliary task data w/ corresponding rotation labels on augmented image
                
                scores_ss = model(aux_x)[1]  # auxilliary tasks for SS head
                scores_main = scores[0]
                
                loss_main = F.cross_entropy(scores_main, aug_y)
                loss_ss = F.cross_entropy(scores_ss, aux_y)
                loss = loss_main + loss_ss                
            else:
                loss = F.cross_entropy(scores, aug_y)

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if (t + 1) % print_every == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t + 1, loss.item()))
                val_acc = check_accuracy(loader_val, model, two_head=two_head, optimizer=optimizer)
                print()
        # Use scheduler to dynamically adjust LR after epoch        
        scheduler.step(val_acc)  
    return check_accuracy(loader_val, model, two_head=two_head, optimizer=optimizer)