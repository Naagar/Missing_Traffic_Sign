import torch
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt

import os
import torch.optim as optim

from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torch.optim.lr_scheduler as lr_scheduler

# from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model

# experiment = Experiment(
#   api_key = "hwPcQ0jxgrwNqfr9MuJ85ORtY",
#   project_name = "general",
#   workspace="naagar"
# )

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
  print('CUDA is not available.  Training on CPU ...')
else:
  print('CUDA is available!  Training on GPU ...')

# src_path = "./datasets/c4mts_2/train/"
# sub_class = os.listdir(src_path)
# lhc = os.listdir(os.path.join(src_path,sub_class[0]))
# gip = os.listdir(os.path.join(src_path,sub_class[1]))
# srl = os.listdir(os.path.join(src_path,sub_class[2]))
# rhc = os.listdir(os.path.join(src_path,sub_class[3]))


# number of subprocesses to use for data loading
num_workers = 10
# how many samples per batch to load
batch_size = 100
# percentage of training set to use as validation
valid_size = 0.1

# number of epochs to train the model
n_epochs = 2


# convert data to a normalized torch.FloatTensor
print('==> Preparing data..')
#Image augmentation is used to train the model
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
#Only the data is normalaized we do not need to augment the test data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# choose the training and test datasets
train_data = datasets.ImageFolder("./datasets/c4mts_2/train/", transform=transform_train)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)

# specify the image classes
classes = ['gap-in-median','left-hand-curve','right-hand-curve','side-road-left']

class BasicBlock(nn.Module):
  expansion = 1
  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(self.expansion*planes)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out

class BottleNeck(nn.Module):
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    super(BottleNeck, self).__init__()
    self.conv1 = nn.Conv2d(in_planes , planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(self.expansion*planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes :
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(self.expansion*planes)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out

class ResNet(nn.Module):
  def __init__(self, block, num_blocks, num_classes=4):
    super(ResNet, self).__init__()
    self.in_planes = 64

    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.linear = nn.Linear(512*4, num_classes)

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    # print(x.shape)
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = F.avg_pool2d(out, (17,30))
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out

# ResNet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
ResNet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

#ResNet34 = ResNet(BasicBlock, [3,4,6,3])
#ResNet50 = ResNet(BottleNeck, [3,4,6,3])
#ResNet101 = ResNet(BottleNeck, [3,4,23,3])
#ResNet152 = ResNet(BottleNeck, [3,8,36,3])

# print(ResNet18)
ResNet18.cuda()

if train_on_gpu:
  ResNet18 = torch.nn.DataParallel(ResNet18)
  cudnn.benchmark = True
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(ResNet18.parameters(), lr=0.0008, momentum=0.9, weight_decay=0.000001)



# Resnet18 = torch.load("./ResNet18.pt") #You can also take pre-trained models on other datasets like imagenet, this will improve accuracy. The model here has not been pre-trained.

valid_loss_min = np.Inf # track change in validation loss


for epoch in range(1, n_epochs+1):
  # keep track of training and validation loss
  train_loss = 0.0
  valid_loss = 0.0

  ###################
  # train the model #
  ###################
  ResNet18.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
      data, target = data.cuda(), target.cuda()
    # clear the gradients of all optimized variables
    optimizer.zero_grad()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = ResNet18(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    # perform a single optimization step (parameter update)
    optimizer.step()
    # update training loss
    train_loss += loss.item()*data.size(0)

  ######################
  # validate the model #
  ######################
  ResNet18.eval()
  for batch_idx, (data, target) in enumerate(valid_loader):
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
      data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = ResNet18(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update average validation loss
    valid_loss += loss.item()*data.size(0)

  # calculate average losses
  train_loss = train_loss/len(train_loader.sampler)
  valid_loss = valid_loss/len(valid_loader.sampler)

  # print training/validation statistics
  print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
      epoch, train_loss, valid_loss))

  # save model if validation loss has decreased
  if valid_loss <= valid_loss_min:
    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
    valid_loss_min,
    valid_loss))
    torch.save(ResNet18.state_dict(), 'ResNet18.pt')
    valid_loss_min = valid_loss


classes = ['gap-in-median','left-hand-curve','right-hand-curve','side-road-left']
sub_classes = ['right-hand-curve', 'gap-in-median', 'side-road-left', 'left-hand-curve']

test_folder = "./datasets/c4mts_2/test/"

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

k=list(range(0,len(os.listdir(test_folder))))
m=0

preprocessed_images = []
predictions_list = []

for filename in os.listdir(test_folder):
    image_path = os.path.join(test_folder, filename)
    image = Image.open(image_path)
    preprocessed_image = transform_test(image)
    preprocessed_images.append(preprocessed_image)
    if len(preprocessed_images) == 10:
        preprocessed_images = torch.stack(preprocessed_images)
        predictions = ResNet18(preprocessed_images)
        for i in range(len(predictions)):
            predicted_class = torch.argmax(predictions[i])
            confidence_score = torch.max(torch.softmax(predictions[i], dim=0))
            print(f"Prediction for Test Image {k[m]+1}: Class {predicted_class.item()}, Confidence: {confidence_score.item()}")
            f=open(f"results_resnet_cls/{k[m]+1}.txt",'w')
            f.write(f"{sub_classes[predicted_class.item()]}, {format(confidence_score.item(),'.2f')} ")
            m+=1
        preprocessed_images = []



















# # -*- coding: utf-8 -*-
# """Copy of C4MTS_TASK2.ipynb

# Automatically generated by Colaboratory.

# Original file is located at
#     https://colab.research.google.com/drive/1TZkn5Rduuq7_0lDTD_FL4C8QM_Zwdagd
# """

# # Commented out IPython magic to ensure Python compatibility.
# # from google.colab import drive
# # drive.mount('/content/drive/')
# # %mkdir /content/drive/MyDrive/myNewfolder
# # %cd /content/drive/MyDrive





# torch.cuda.empty_cache()

# # check if CUDA is available
# train_on_gpu = torch.cuda.is_available()

# if not train_on_gpu:
#   print('CUDA is not available.  Training on CPU ...')
# else:
#   print('CUDA is available!  Training on GPU ...')

# # Commented out IPython magic to ensure Python compatibility.
# # %matplotlib inline

# src_path = "./datasets/c4mts_2/train/"
# sub_class = os.listdir(src_path)
# lhc = os.listdir(os.path.join(src_path,sub_class[0]))
# gip = os.listdir(os.path.join(src_path,sub_class[1]))
# srl = os.listdir(os.path.join(src_path,sub_class[2]))
# rhc = os.listdir(os.path.join(src_path,sub_class[3]))

# # fig = plt.figure(figsize=(100,100))
# # path = os.path.join(src_path,sub_class[0])
# # for i in range(0,1):
# #     plt.subplot(240 + 1 + i)
# #     img = plt.imread(os.path.join(path,lhc[0]))
# #     plt.imshow(img, cmap=plt.get_cmap('gray'))
# # path = os.path.join(src_path,sub_class[1])
# # for i in range(0,1):
# #     plt.subplot(240 + 2 + i)
# #     img = plt.imread(os.path.join(path,gip[0]))
# #     plt.imshow(img, cmap=plt.get_cmap('gray'))
# # path = os.path.join(src_path,sub_class[2])
# # for i in range(0,1):
# #     plt.subplot(240 + 3 + i)
# #     img = plt.imread(os.path.join(path,srl[0]))
# #     plt.imshow(img, cmap=plt.get_cmap('gray'))
# # path = os.path.join(src_path,sub_class[3])
# # for i in range(0,1):
# #     plt.subplot(240 + 4 + i)
# #     img = plt.imread(os.path.join(path,rhc[0]))
# #     plt.imshow(img, cmap=plt.get_cmap('gray'))



# # number of subprocesses to use for data loading
# num_workers = 4
# # how many samples per batch to load
# batch_size = 16
# # percentage of training set to use as validation
# valid_size = 0.1

# # number of epochs to train the model
# n_epochs = 12

# # convert data to a normalized torch.FloatTensor
# print('==> Preparing data..')
# #Image augmentation is used to train the model
# transform_train = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     transforms.RandomResizedCrop(256),
#     # transforms.RandomHorizontalFlip(),
#     transforms.RandomHorizontalFlip(p=0.5),            ###               Done, vertival flip , random_rotate , brightness and contrast 
#     transforms.RandomVerticalFlip(p=0.5), 
#     # transforms.ColorJitter(brightness=0.10,saturation=0.090,contrast=0.09, hue=0.09),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[1.0817, 1.1146, 0.9792], std=[0.8482, 0.9573, 1.1026])
# ])
# #Only the data is normalaized we do not need to augment the test data
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# print('==> Load dataset and split')

# # choose the training and test datasets
# train_data = datasets.ImageFolder(src_path, transform=transform_train)
# # test_data = datasets.ImageFolder("./datasets/c4mts_2/test/", transform=transform_test)
# val_data = datasets.ImageFolder("./datasets/c4mts_2/val/", transform=transform_train)



# # obtain training indices that will be used for validation
# num_train = len(train_data)
# train_indices = list(range(num_train))
# # np.random.shuffle(indices)
# # split = int(np.floor(valid_size * num_train))
# # train_idx, valid_idx = indices[split:], indices[:split]

# num_val = len(val_data)
# valid_idx = list(range(num_val))

# # define samplers for obtaining training and validation batches
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(valid_idx)
# print('data split done.')
# # prepare data loaders (combine dataset and sampler)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
#     sampler=train_sampler, num_workers=num_workers)
# valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
#     sampler=valid_sampler, num_workers=num_workers)
# # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# # specify the image classes
# classes = ['gap-in-median','left-hand-curve','right-hand-curve','side-road-left']

# print('==> Definig model')


# class BasicBlock(nn.Module):
#   expansion = 1
#   def __init__(self, in_planes, planes, stride=1):
#     super(BasicBlock, self).__init__()
#     self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#     self.bn1 = nn.BatchNorm2d(planes)
#     self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#     self.bn2 = nn.BatchNorm2d(planes)

#     self.shortcut = nn.Sequential()
#     if stride != 1 or in_planes != self.expansion*planes:
#       self.shortcut = nn.Sequential(
#           nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#           nn.BatchNorm2d(self.expansion*planes)
#       )

#   def forward(self, x):
#     out = F.relu(self.bn1(self.conv1(x)))
#     out = self.bn2(self.conv2(out))
#     out += self.shortcut(x)
#     out = F.relu(out)
#     return out

# class BottleNeck(nn.Module):
#   expansion = 1

#   def __init__(self, in_planes, planes, stride=1):
#     super(BottleNeck, self).__init__()
#     self.conv1 = nn.Conv2d(in_planes , planes, kernel_size=1, bias=False)
#     self.bn1 = nn.BatchNorm2d(planes)
#     self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#     self.bn2 = nn.BatchNorm2d(planes)
#     self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#     self.bn3 = nn.BatchNorm2d(self.expansion*planes)

#     self.shortcut = nn.Sequential()
#     if stride != 1 or in_planes != self.expansion*planes :
#       self.shortcut = nn.Sequential(
#           nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#           nn.BatchNorm2d(self.expansion*planes)
#       )

#   def forward(self, x):
#     out = F.relu(self.bn1(self.conv1(x)))
#     out = F.relu(self.bn2(self.conv2(out)))
#     out = self.bn3(self.conv3(out))
#     out += self.shortcut(x)
#     out = F.relu(out)
#     return out

# class ResNet(nn.Module):
#   def __init__(self, block, num_blocks, num_classes=4):
#     super(ResNet, self).__init__()
#     self.in_planes = 64

#     self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     self.bn1 = nn.BatchNorm2d(64)
#     self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#     self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#     self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#     self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#     self.linear = nn.Linear(512*4, num_classes)

#   def _make_layer(self, block, planes, num_blocks, stride):
#     strides = [stride] + [1]*(num_blocks-1)
#     layers = []
#     for stride in strides:
#       layers.append(block(self.in_planes, planes, stride))
#       self.in_planes = planes * block.expansion
#     return nn.Sequential(*layers)

#   def forward(self, x):
#     # print(x.shape)
#     out = F.relu(self.bn1(self.conv1(x)))
#     out = self.layer1(out)
#     out = self.layer2(out)
#     out = self.layer3(out)
#     out = self.layer4(out)
#     out = F.avg_pool2d(out, (17,30))
#     out = out.view(out.size(0), -1)
#     out = self.linear(out)
#     return out


# ## choose model 
# # model_ft = models.alexnet(pretrained=True)
# # model_ft = models.squeezenet1_0(pretrained=True)
# # model_ft = models.vgg16(pretrained=True)
# # model_ft = models.densenet161(pretrained=True)
# # model_ft = models.inception_v3(pretrained=True)
# # model_ft = models.googlenet(pretrained=True)
# # model_ft = models.shufflenet_v2_x1_0(pretrained=True)
# # model_ft = models.mobilenet_v2(pretrained=True)
# # model_ft = models.resnext50_32x4d(pretrained=True)
# # model_ft = models.wide_resnet50_2(pretrained=True)
# # model_ft = models.mnasnet1_0(pretrained=True)
# # model_ft = models.resnet18(pretrained=True)
# ResNet18 = ResNet(BasicBlock, [1,1,1,1])  #ResNet18
# # ResNet18 = ResNet(BasicBlock, [3,4,6,3])    #ResNet34
# # ResNet18 = ResNet(BottleNeck, [3,4,6,3])   #ResNet50
# #ResNet101 = ResNet(BottleNeck, [3,4,23,3]) #ResNet101
# #ResNet152 = ResNet(BottleNeck, [3,8,36,3]) #ResNet152

# # print(ResNet18)

# if train_on_gpu:
#   ResNet18 = torch.nn.DataParallel(ResNet18)
#   cudnn.benchmark = True

# import torch.optim as optim
# # specify loss function (categorical cross-entropy)
# criterion = nn.CrossEntropyLoss()

# # lr rate schedular 
# def lr_lambda(epoch):
#     # LR to be 0.1 * (1/1+0.01*epoch)
#     base_lr = 0.1
#     factor = 0.01
#     return base_lr/(1+factor*epoch)

# # specify optimizer
# optimizer = optim.SGD(ResNet18.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00001)
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)




# print('==> Load model')
# # Resnet18 = torch.load("ResNet18.pt")

# valid_loss_min = np.Inf # track change in validation loss


# for epoch in range(1, n_epochs+1):
#   # keep track of training and validation loss
#   train_loss = 0.0
#   valid_loss = 0.0

#   ###################
#   # train the model #
#   ###################
#   ResNet18.train()
#   for batch_idx, (data, target) in enumerate(train_loader):
#     # move tensors to GPU if CUDA is available
#     if train_on_gpu:
#       data, target = data.cuda(), target.cuda()
#     # clear the gradients of all optimized variables
#     optimizer.zero_grad()
#     # forward pass: compute predicted outputs by passing inputs to the model
#     output = ResNet18(data)
#     # calculate the batch loss
#     loss = criterion(output, target)
#     # backward pass: compute gradient of the loss with respect to model parameters
#     loss.backward()
#     # perform a single optimization step (parameter update)
#     optimizer.step()
#     # update training loss
#     train_loss += loss.item()*data.size(0)

#   ######################
#   # validate the model #
#   ######################
#   ResNet18.eval()
#   for batch_idx, (data, target) in enumerate(valid_loader):
#     # move tensors to GPU if CUDA is available
#     if train_on_gpu:
#       data, target = data.cuda(), target.cuda()
#     # forward pass: compute predicted outputs by passing inputs to the model
#     output = ResNet18(data)
#     # calculate the batch loss
#     loss = criterion(output, target)
#     # update average validation loss
#     valid_loss += loss.item()*data.size(0)

#   # calculate average losses
#   train_loss = train_loss/len(train_loader.sampler)
#   valid_loss = valid_loss/len(valid_loader.sampler)

#   # print training/validation statistics
#   print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
#       epoch, train_loss, valid_loss))

#   # save model if validation loss has decreased
#   if valid_loss <= valid_loss_min:
#     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
#     valid_loss_min,
#     valid_loss))
#     torch.save(ResNet18.state_dict(), 'ResNet18.pt')
#     valid_loss_min = valid_loss

# print('==> training finished')
# ResNet18.load_state_dict(torch.load('ResNet18.pt'))



# classes = ['gap-in-median','left-hand-curve','right-hand-curve','side-road-left']
# sub_classes = ['right-hand-curve', 'gap-in-median', 'side-road-left', 'left-hand-curve']

# test_folder = "./datasets/task2test/"
# batch_size = 100

# # transform_test = transforms.Compose([
# #     transforms.ToTensor(),
# #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
# # ])

# ## Predction/ testing
# # list of test images
# k=list(range(0,len(os.listdir(test_folder))))
# m=0

# preprocessed_images = []
# predictions_list = []

# for filename in os.listdir(test_folder):
#     image_path = os.path.join(test_folder, filename)
#     image = Image.open(image_path)
#     preprocessed_image = transform_test(image)
#     preprocessed_images.append(preprocessed_image)
#     if len(preprocessed_images) == batch_size:
#         preprocessed_images = torch.stack(preprocessed_images)
#         predictions = ResNet18(preprocessed_images)
#         for i in range(len(predictions)):
#             predicted_class = torch.argmax(predictions[i])
#             confidence_score = torch.max(torch.softmax(predictions[i], dim=0))
#             print(f"Prediction for Test Image {k[m]+1}: Class {predicted_class.item()}, Confidence: {confidence_score.item()}")
#             f=open(f"{k[m]+1}.txt",'w')
#             f.write(f"{sub_classes[predicted_class.item()]}, {format(confidence_score.item(),'.2f')} ")
#             m+=1
#         preprocessed_images = []
