from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchdependencies import train_model, set_parameter_requires_grad, initialize_model, RoadDataset, make_weights_for_balanced_classes


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "densenet"

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 31

# Size of the image
input_size = 224

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

torch.manual_seed(42)

print(f"Initializing the {model_name} model...")

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: RoadDataset(mode=x, transform=data_transforms[x]) for x in ['train', 'val']}

#weights, weight_per_class = make_weights_for_balanced_classes(image_datasets['train'], num_classes)                                                                
#weights = torch.DoubleTensor(weights)
#print(f"Class-wise weights calculated for sampler: {weight_per_class}")

# Create training and validation dataloaders
dataloaders_dict = {
#    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, num_workers=4, sampler=torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)),
    'val': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, num_workers=4, shuffle=False)
}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# # Train and evaluate
# model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, device=device, is_inception=(model_name=="inception"))
# 
# torch.save(model_ft, f"/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/road_classification/torch_logs/{model_name}")
# torch.save({
#             'epoch': num_epochs,
#             'model_state_dict': model_ft.state_dict(),
#             'optimizer_state_dict': optimizer_ft.state_dict(),
#             }, f"/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/road_classification/torch_logs/{model_name}.pt")

checkpoint = torch.load(f"/deep/u/ayushsn/satellite-pixel-synthesis-pytorch/road_classification/torch_logs/{model_name}_1.pt")
model_ft.load_state_dict(checkpoint['model_state_dict'])
optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
#metrics = [
#    Accuracy(task='multiclass', num_classes=num_classes, average='micro'), 
#    F1Score(task='multiclass', num_classes=num_classes, average='micro'), 
#    Precision(task='multiclass', num_classes=num_classes, average='micro'), 
#    Recall(task='multiclass', num_classes=num_classes, average='micro')
#]
metrics = [BinaryAccuracy(), BinaryF1Score(), BinaryPrecision(), BinaryRecall()]
metric_names = ["Val Accuracy", "Val F1", "Val Precision", "Val Recall"]
P = []
T = []
for inputs, labels in dataloaders_dict['val']:
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer_ft.zero_grad()
    with torch.set_grad_enabled(False):
        outputs = model_ft(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        P.append(preds.to('cpu'))
        T.append(labels.data.to('cpu'))
predictions = torch.cat(P, 0)
targets = torch.cat(T, 0)
for i in range(4):
    print(f"{metric_names[i]}: {metrics[i](predictions, targets)}")
