# Self-Supervised Test-Time Training

## Introduction
This project intends to develop a model that can use testing data to tune its own parameters using self-supervised learning on data being fed in during test time. This will allow the model to continue learning after being deployed and become more robust, learning shifting distributions in the input data. 

The results are already loaded in the main.ipynb notebook, but it can be ran so long as all the packages are already installed. It will automatically download the dataset, train the model, and display all the results of the experiments. 

I use supervised learning to create models that can classify new inputs after it learns on training data, but the model is limited by how much data can be gathered and classified before deploying the model. There can also be underlying errors with how training data is collected, i.e. data that inaccurately models the real world or mislabeled data that propagates to the end model. These factors can be accounted for without having to seriously expand our training set if we had a way to implement training once the model has already been deployed. Under this paradigm, I modified a supervised learning model and implemented training at test-time, which will be robust to changes in distributions of data. 

To do this, I implemented a self-supervised task, which consisted of rotating an image 90 degrees. Although a model performing this operation wouldn’t know what the ground truth label is for classification, it can generate a label based on the random rotation and use the accuracy on that task to optimize a loss function. I coupled this with a branching ResNet, which had three components: a shared feature extractor, a main prediction head, and a self-supervised learning head, to continuously update its parameters on this self-supervised task while retaining the ability to classify images.

My results find that the performance of a model that can train at test-time has similar performance to a regular ResNet implementation on clean data, and is significantly less erroneous on various types of common image noise/distortions.

## Method
Reimplementing methods from the research paper “Test-Time Training with Self-Supervision for Generalization under Distribution Shifts” (Sun et al., 2020), I propose using a deep network that follows a tree-like model architecture. The input will be fed into a feature extractor branch, which is shared between two other branches down the line: a self-supervised branch that trains on the previously described 90 degree rotation task, and a main classification branch that will predict test data. 

At training time, both heads will produce their own losses that will be summed and optimized over, and no test-time training will occur on the validation set. 
The reasoning for separating these two branches is to allow the feature extractor to be tuned, as when distributions in the data shift, the features will need to be updated to maintain accuracy. However, the model can’t tune all its parameters on the auxiliary rotation task, or it risks catastrophically forgetting the actual image classification task. By splitting the network into two heads, the idea is that the main head will remember the image classification task, while the self-supervised network learns the rotation classification task. All the while, the feature extractor updates during self-supervised learning.

While training at test time, the model will generate data augmented copies of the original test image batch. This is using standard data augmentation techniques, random rescaled cropping and horizontal flipping. This will generate n batches for n new data augmented examples, and the model will train on each batch sequentially, before finally predicting the image. This allows the model to partially learn the distribution immediately, and mostly learn it over time.

For the deep network itself, I decided to use group norm after experimenting with both batch norm and instance norm, where it worked well at keeping the model robust similarly to in Sun’s paper. The feature extractor consists of a convolutional layer that feeds into two residual layers, each defined by a norm-relu-conv structure. Each layer here downsamples the image using max pooling. After that, the main head and self-supervised head share a similar architecture, with both using two residual layers without downsampling, into a global average pool, ending with a fully connected layer. 

To tune the test-time training hyperparameters, I split the validation set into a clean validation set used in the training phase pre-deployment, and a dirty validation set that would be modified with various corruptions to see how well the model performs when doing its self-supervised learning.

