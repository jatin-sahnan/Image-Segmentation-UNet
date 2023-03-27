#!/usr/bin/env python
# coding: utf-8

# ## Finetuned UNet model with ResNet backbone for image segmentation on 24 pathology test images, and reporting accuracy for each class.

# In[1]:


#Importing Libraries

import os
import cv2
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from PIL import Image


# In[3]:


if not torch.cuda.is_available():
  raise Exception("GPU not availalbe. CPU training will be too slow.")
print("device name", torch.cuda.get_device_name(0))


# In[4]:


get_ipython().system('git clone https://github.com/usuyama/pytorch-unet.git')
get_ipython().run_line_magic('cd', 'pytorch-unet')


# In[5]:


get_ipython().system('ls')


# ### Train Dataset

# In[6]:


class PathologyDataset(Dataset):
    def __init__(self,image_dir,mask_dir):
        self.input_images = sorted(glob.glob(image_dir+'/*'))
        self.input_masks = sorted(glob.glob(mask_dir+'/*'))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        
    def __len__(self):
        
        return len(self.input_images)
    
    def __getitem__(self, idx):
        
        x = self.input_images[idx]
        y = self.input_masks[idx]
        
        image = cv2.imread(x)[:,:,::-1] #Converting BGR to RGB
        image = cv2.resize(image, (224,224))
        image = torch.tensor(image/255)
        image = image.permute(2,0,1)
        image = self.normalize(image)
        
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE) #Converting to grayscale
        mask = cv2.resize(mask, (224,224))
        mask = torch.tensor(mask/255)
        mask = np.expand_dims(mask,axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        
        return [image,mask]


# In[7]:


# Define the data directories
data_dir = "/content/drive/My Drive/pathologyData/train"
image_dir = data_dir + "/images"
mask_dir = data_dir + "/masks"

train_dataset = PathologyDataset(image_dir,mask_dir)
val_dataset = PathologyDataset(image_dir,mask_dir)

image_datasets = {'train': train_dataset, 'val': val_dataset}


# In[8]:


# loading the newly created datasets
batch_size=10
dataloaders = {'train' : DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=0),
               'val' : DataLoader(val_dataset, batch_size = batch_size, shuffle=True, num_workers=0)}


# In[9]:


# Get a batch of training data
inputs, masks = next(iter(dataloaders['train']))

print(inputs.shape, masks.shape)

def reverse_transform(inp):
  inp = inp.cpu().numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)
  inp = (inp * 255).astype(np.uint8)

  return inp

plt.imshow(reverse_transform(inputs[3]))
plt.imshow(reverse_transform(masks[3]))


# ### Test Dataset

# In[10]:


# Define the data directories
data_dir1 = "/content/drive/My Drive/pathologyData/test"
image_dir1 = data_dir + "/images"
mask_dir1 = data_dir + "/masks"

test_dataset = PathologyDataset(image_dir1,mask_dir1)


# In[11]:


# loading the test dataset
batch_size=4
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=0)


# In[12]:


# Get a batch of test data
inputs, masks = next(iter(test_dataloader))

print(inputs.shape, masks.shape)

def reverse_transform(inp):
  inp = inp.cpu().numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)
  inp = (inp * 255).astype(np.uint8)

  return inp

plt.imshow(reverse_transform(inputs[3]))


# ### Define Unet

# In[13]:


def convrelu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.ReLU(inplace=True),
  )


class ResNetUNet(nn.Module):
  def __init__(self, n_class):
    super().__init__()

    self.base_model = torchvision.models.resnet18(pretrained=True)
    self.base_layers = list(self.base_model.children())

    self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
    self.layer0_1x1 = convrelu(64, 64, 1, 0)
    self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
    self.layer1_1x1 = convrelu(64, 64, 1, 0)
    self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
    self.layer2_1x1 = convrelu(128, 128, 1, 0)
    self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
    self.layer3_1x1 = convrelu(256, 256, 1, 0)
    self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
    self.layer4_1x1 = convrelu(512, 512, 1, 0)

    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
    self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
    self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
    self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

    self.conv_original_size0 = convrelu(3, 64, 3, 1)
    self.conv_original_size1 = convrelu(64, 64, 3, 1)
    self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

    self.conv_last = nn.Conv2d(64, n_class, 1)

  def forward(self, input):
    x_original = self.conv_original_size0(input)
    x_original = self.conv_original_size1(x_original)

    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)
    layer3 = self.layer3(layer2)
    layer4 = self.layer4(layer3)

    layer4 = self.layer4_1x1(layer4)
    x = self.upsample(layer4)
    layer3 = self.layer3_1x1(layer3)
    x = torch.cat([x, layer3], dim=1)
    x = self.conv_up3(x)

    x = self.upsample(x)
    layer2 = self.layer2_1x1(layer2)
    x = torch.cat([x, layer2], dim=1)
    x = self.conv_up2(x)

    x = self.upsample(x)
    layer1 = self.layer1_1x1(layer1)
    x = torch.cat([x, layer1], dim=1)
    x = self.conv_up1(x)

    x = self.upsample(x)
    layer0 = self.layer0_1x1(layer0)
    x = torch.cat([x, layer0], dim=1)
    x = self.conv_up0(x)

    x = self.upsample(x)
    x = torch.cat([x, x_original], dim=1)
    x = self.conv_original_size2(x)

    out = self.conv_last(x)

    return out


# In[14]:


## Instantiate Unet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

model = ResNetUNet(1) #As there are two classes
model = model.to(device)


# In[15]:


model


# In[16]:


from torchsummary import summary
summary(model, input_size=(3, 224, 224))


# In[17]:


from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss

checkpoint_path = "checkpoint.pth"

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.float().to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'train':
              scheduler.step()
              for param_group in optimizer.param_groups:
                  print("LR", param_group['lr'])

            # save the model weights
            if phase == 'val' and epoch_loss < best_loss:
                print(f"saving best model to {checkpoint_path}")
                best_loss = epoch_loss
                torch.save(model.state_dict(), checkpoint_path)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(torch.load(checkpoint_path))
    return model


# In[18]:


## Training the model

from torch.optim import lr_scheduler
import time

num_class = 1
model = ResNetUNet(num_class).to(device)

# freeze backbone layers
for l in model.base_layers:
  for param in l.parameters():
    param.requires_grad = False

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

#Decays the learning rate of each parameter group by gamma every step_size epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.1)

model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=20)


# ### Evaluating the model

# In[19]:


def prepare_plot(origImage, origMask, predMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(reverse_transform(origImage))
    ax[1].imshow(reverse_transform(origMask))
    ax[2].imshow(reverse_transform(predMask))
    # set the titles of the subplots
    ax[0].set_title("Original Image")
    ax[1].set_title("Ground Truth Image")
    ax[2].set_title("Predicted Mask Image")
    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()


# In[23]:


model.eval()

num_correct = 0
num_pixels = 0
img_count = 1
class1_count = 0
class0_count = 0
class1_correct = 0
class0_correct = 0

for inputs, labels in test_dataloader:
    inputs = inputs.float().to(device)
    labels = labels.to(device)
    
    pred = model(inputs)
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    
    #Set predicted values > 0.5 to 1 (Foreground) and predicted values < 0.5 as 0 (Background)
    
    pred = pred > 0.5
    pred = pred.astype(np.uint8)
    pred = torch.tensor(pred)
    
    #Set true mask values > 0.5 to 1 and true mask values < 0.5 as 0
    
    new_label = labels
    new_label = new_label.data.cpu().numpy()
    new_label = new_label>0.5
    new_label = new_label.astype(np.uint8)
    new_label = torch.tensor(new_label)
    
    ct_0 = torch.sum((new_label==pred)*(new_label==0))
    class0_correct = class0_correct + ct_0
    
    ct_1 = torch.sum((new_label==pred)*(new_label==1))
    class1_correct = class1_correct + ct_1
    
    mask_indices_1 = torch.nonzero(new_label==1)
    class1_count = class1_count + mask_indices_1.shape[0]
    
    mask_indices_0 = torch.nonzero(new_label==0)
    class0_count = class0_count + mask_indices_0.shape[0]
    
    num_correct += (pred == new_label).sum()
    num_pixels += torch.numel(pred)
    
    img_count = img_count+1
    
    for k in range(len(inputs)):
        prepare_plot(inputs[k], new_label[k], pred[k])
    
acc = (num_correct/num_pixels*100).numpy()
print("\n Overall Test Accuracy: ",np.around(acc, 2),"%")

class1_acc = (class1_correct/class1_count*100).numpy()
print("\n Class 1 (Foreground) Test Accuracy: ",np.around(class1_acc, 2),"%")
class0_acc = (class0_correct/class0_count*100).numpy()
print("\n Class 0 (Background) Test Accuracy: ",np.around(class0_acc, 2),"%")


# In[20]:




