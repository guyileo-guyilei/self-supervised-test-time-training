#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import matplotlib.pyplot as plt


# ## Setup

# In[2]:


# General parameters
RUN_TEST = True  # If done with tuning model, set to True
USE_GPU = True
batch_size = 64


# In[3]:


dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('using device:', device)


# In[4]:


from torchvision import datasets

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)


# In[5]:


# Dataloaders
NUM_VAL = 10000
NUM_TOTAL = 60000

# To tune test-time training, I split the training set into two validation sets, one for clean validation 
# and one for test-time training validation

# Validation set used during training to tune network hyperparameters
loader_val_clean = DataLoader(train_data, batch_size=batch_size, num_workers=4,
                          sampler=sampler.SubsetRandomSampler(range(NUM_VAL)))

# Validation set used during TTT to tune self-supervised hyperparameters
loader_val_ttt = DataLoader(train_data, batch_size=batch_size, num_workers=4,
                          sampler=sampler.SubsetRandomSampler(range(NUM_VAL, NUM_VAL*2)))

loader_train = DataLoader(train_data, batch_size=batch_size, num_workers=4,
                          sampler=sampler.SubsetRandomSampler(range(NUM_VAL*2, NUM_TOTAL)))

loader_test = DataLoader(test_data, batch_size=batch_size, num_workers=4)


# ## Training and Validation

# In[6]:


from training_testing.training import train

# Training parameters
retrain = True
lr = 3e-3
weight_decay = 5e-4
epochs = 10
momentum = 0.9


# In[7]:


# Training the SSResNet
from models.SSResNet import SSResNet

if retrain:  # Prevent from retraining when not needed
    model_ss = SSResNet().to(device=device)
    optimizer = optim.SGD(model_ss.parameters(),lr=lr,momentum=momentum, weight_decay=weight_decay)
    train(model_ss,optimizer,loader_train,loader_val_clean,epochs=epochs,two_head=True)


# In[8]:


# Training the ResNet baseline
from models.ResNet import ResNet

if retrain:  # Prevent from retraining when not needed
    model_resnet = ResNet().to(device=device)
    optimizer = optim.SGD(model_resnet.parameters(),lr=lr,momentum=momentum, weight_decay=weight_decay)
    train(model_resnet,optimizer,loader_train,loader_val_clean,epochs=epochs,two_head=False)


# In[9]:


# Validate on some gaussian noise to test robustness
from utils.transforms import * 

gaussian_noise = GaussianNoise(std=0.25)


# In[40]:


# SSResNet TTT Validation on both clean and noisy data
from training_testing.testing import check_accuracy
import copy

print('Validating on clean data...')
model_copy_ss = copy.deepcopy(model_ss)

ss_optimizing_params = list(model_copy_ss.shared_branch.parameters()) + list(model_copy_ss.ss_head.parameters())
optimizer = optim.SGD(ss_optimizing_params,lr=lr,momentum=0, weight_decay=0)

check_accuracy(loader_val_ttt,model_copy_ss,two_head=True,ttt=True,optimizer=optimizer)
print()


print('Validating on noisy data...')
model_copy_ss = copy.deepcopy(model_ss)

ss_optimizing_params = list(model_copy_ss.shared_branch.parameters()) + list(model_copy_ss.ss_head.parameters())
optimizer = optim.SGD(ss_optimizing_params,lr=lr,momentum=0, weight_decay=0)

check_accuracy(loader_val_ttt,model_copy_ss,two_head=True,ttt=True,optimizer=optimizer,transform=gaussian_noise)


# In[41]:


# SSResNet (no TTT) Validation
model_copy_ss = copy.deepcopy(model_ss)

print('Validating on clean data...')
check_accuracy(loader_val_ttt,model_copy_ss,two_head=True)
print()

print('Validating on noisy data...')
check_accuracy(loader_val_ttt,model_copy_ss,two_head=True,transform=gaussian_noise)


# In[42]:


# ResNet baseline Validation
optimizer = optim.SGD(model_resnet.parameters(),lr=lr,momentum=0.9, weight_decay=5e-4)

print('Validating on clean data...')
check_accuracy(loader_val_ttt,model_resnet,two_head=False,optimizer=optimizer)
print()

print('Validating on noisy data...')
check_accuracy(loader_val_ttt,model_resnet,two_head=False,optimizer=optimizer,transform=gaussian_noise)


# In[50]:


# Visualize auxiliary task, labels 0-1-2-3 correspond to 0-90-180-270 degree CCW rotations
from utils.RandomRotation90deg import *

batch,_ = next(iter(loader_train))
auxiliary_task = RandomRotation90deg()
batch,labels = auxiliary_task(batch)

fig, ax = plt.subplots(2, 2)
for i in range(2):
    for j in range(2):
        ax[i,j].imshow(batch[2*i+j].squeeze(),cmap='gray')
        ax[i,j].set_title(f'Label: {labels[2*i+j]}')
        ax[i,j].axis('off')
        
fig.suptitle("Example images the auxiliary task trains on")
plt.show()


# ## Testing and Experiments

# In[13]:


# After validation finishes, do final plots either with appropriate set
if RUN_TEST:
    loader = loader_test
else:
    loader = loader_val_ttt


# In[14]:


# Increasing gaussian noise as batches come in
from training_testing.testing import test_increasing_noise

model_copy_ss = copy.deepcopy(model_ss)
ss_optimizing_params = list(model_copy_ss.shared_branch.parameters()) + list(model_copy_ss.ss_head.parameters())
optimizer = optim.SGD(ss_optimizing_params,lr=lr,momentum=0, weight_decay=0)

max_noise = .5
transformClass = GaussianNoise

errors_joint = test_increasing_noise(loader,model_copy_ss,two_head=True,ttt=False,optimizer=optimizer,transformClass=transformClass,max_noise=max_noise)
errors_ttt = test_increasing_noise(loader,model_copy_ss,two_head=True,ttt=True,optimizer=optimizer,transformClass=transformClass,max_noise=max_noise)
errors_base = test_increasing_noise(loader,model_resnet,optimizer=optimizer,transformClass=transformClass,max_noise=max_noise,sample=True)

plt.plot(errors_base,color='k')
plt.plot(errors_joint,color='c')
plt.plot(errors_ttt,color='r')
plt.legend(["Object Recognition Only","Joint Training","TTT"])
plt.xlabel("Batch Number")
plt.ylabel("Cumulative Error")
plt.title("Cumulative Error vs. Batch Number for Gaussian Noise")
plt.grid(True)
plt.show()


# In[15]:


# Increasing impulse noise as batches come in
model_copy_ss = copy.deepcopy(model_ss)
ss_optimizing_params = list(model_copy_ss.shared_branch.parameters()) + list(model_copy_ss.ss_head.parameters())
optimizer = optim.SGD(ss_optimizing_params,lr=lr,momentum=0, weight_decay=0)

max_noise = .5
transformClass = ImpulseNoise

errors_joint = test_increasing_noise(loader,model_copy_ss,two_head=True,ttt=False,optimizer=optimizer,transformClass=transformClass,max_noise=max_noise)
errors_ttt = test_increasing_noise(loader,model_copy_ss,two_head=True,ttt=True,optimizer=optimizer,transformClass=transformClass,max_noise=max_noise)
errors_base = test_increasing_noise(loader,model_resnet,optimizer=optimizer,transformClass=transformClass,max_noise=max_noise,sample=True)

plt.plot(errors_base,color='k')
plt.plot(errors_joint,color='c')
plt.plot(errors_ttt,color='r')
plt.legend(["Object Recognition Only","Joint Training","TTT"])
plt.xlabel("Batch Number")
plt.ylabel("Error")
plt.title("Cumulative Error vs. Batch Number for Impulse Noise")
plt.grid(True)
plt.show()


# In[16]:


# Visualize transforms and get accuracy of classifiers on noisy data
transform_list = [Identity,GaussianNoise,ImpulseNoise,Pixelate,Defocus,JPEGCompression]
accuracy_per_classifier = []

for noiseTransform in transform_list:
    model_copy_ss = copy.deepcopy(model_ss)
    ss_optimizing_params = list(model_copy_ss.shared_branch.parameters()) + list(model_copy_ss.ss_head.parameters())
    optimizer = optim.SGD(ss_optimizing_params,lr=lr,momentum=0, weight_decay=0)
    
    noise_transform = noiseTransform()
    acc_base = check_accuracy(loader,model_resnet,printing=False,transform=noise_transform)
    acc_joint = check_accuracy(loader,model_copy_ss,two_head=True,printing=False,transform=noise_transform)
    acc_ttt = check_accuracy(loader,model_copy_ss,two_head=True,ttt=True,printing=False,transform=noise_transform,optimizer=optimizer)
    
    accuracy_per_classifier.append((acc_base,acc_joint,acc_ttt))

    # Display example image
    x,_ = next(iter(loader))
    img = x[0]
    noisy_img = noise_transform(img)
    img = img.squeeze()
    noisy_img = noisy_img.squeeze()
    
    fig, ax = plt.subplots(1, 2)  # 1 row, 2 columns
    ax[0].imshow(img,cmap='gray')
    ax[0].axis('off')
    ax[0].set_title("Clean")

    ax[1].imshow(noisy_img,cmap='gray')
    ax[1].axis('off')
    ax[1].set_title(f"{noiseTransform.__name__} transform")
    plt.show()
    
accuracy_per_classifier = np.array(accuracy_per_classifier)


# In[17]:


# Plot error each classifier has on high intensity noise
errors_per_classifier = 1 - accuracy_per_classifier
categories = [Class.__name__ for Class in transform_list]  # c categories
results = ['Object Recognition Only','Joint Training', 'TTT']  # n results per category

x = np.arange(len(categories))
width = 0.2

fig, ax = plt.subplots(figsize=(8, 5))

# Plot each result as a separate bar within each category
for i, result in enumerate(results):
    ax.bar(x + i * width, errors_per_classifier[:, i], width, label=result)  

ax.set_xticks(x + width)  # Center labels
ax.set_xticklabels(categories)
ax.set_ylabel('Error')
ax.set_title('Error Comparison on High Intensity Corruptions')
ax.legend(title='Classifiers')

plt.show()


# In[ ]:




