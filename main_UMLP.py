# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:22:42 2022

@author: WiCi
"""

from __future__ import print_function
import argparse
from math import log10
from os.path import exists, join, basename
from os import makedirs, remove

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model_UMLP import *
import torch.utils.data as Data
import numpy as np
import math
import matplotlib.pyplot as plt

# Partial Parameter initialization
parser = argparse.ArgumentParser(description='U-MLP')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', default=True,action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialization randn seed
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

# load model
model = channel_est().to(device)
model.to(device)
print (model)

# Statistical model parameters and FLOPs
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('parameters_count:',count_parameters(model))

from thop import profile
inputest1 = torch.randn(1, 2, 512, 32).cuda()
flops, params = profile(model, inputs=(inputest1,))
from thop import clever_format
flops, params = clever_format([flops, params], "%.3f")
print('flops: ', flops, 'params: ', params)

# loading Training data and Validation data
# The desired data path need to be swtiched
import h5py
path="inHmix_73_32_512_4users_64pilot.mat"

with h5py.File(path, 'r') as file:
    train_h = np.transpose(np.array(file['output_da']))
    train_h = train_h.transpose([0,3,1,2])
    test_h = np.transpose(np.array(file['output_da_test']))
    test_h = test_h.transpose([0,3,1,2])
#    test_snr = np.transpose(np.array(file['Testsnr']))
with h5py.File(path, 'r') as file:
    train_y = np.transpose(np.array(file['input_da']))
    train_y = train_y.transpose([0,3,1,2])
    test_y = np.transpose(np.array(file['input_da_test']))
    test_y = test_y.transpose([0,3,1,2])
    
X_train=torch.Tensor(train_y)
X_val=torch.Tensor(test_y)
y_train=torch.Tensor(train_h)
y_val=torch.Tensor(test_h)

train_set = Data.TensorDataset(X_train, y_train) 
val_set = Data.TensorDataset(X_val, y_val) 

# clear the occupied memory
del train_h
del test_h
del train_y
del test_y

batchsize = 16
train_loader = torch.utils.data.DataLoader(train_set, batch_size= batchsize, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size= batchsize, shuffle=True)

# definition of the loss function
def CharbonnierLoss(predict, target):
    return torch.mean(torch.sqrt(torch.pow((predict.cpu().double()-target.cpu().double()), 2) + 1e-3)) # epsilon=1e-3

# checkpoint setup
def checkpoint(epoch):
    if not exists(opt.checkpoint):
        makedirs(opt.checkpoint)
    model_out_path = "model/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    
# Learning rate decay schedule
def adjust_learning_rate(optimizer, epoch,learning_rate_init,learning_rate_final):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    epochs =opt.nEpochs
    lr = learning_rate_final + 0.5*(learning_rate_init-learning_rate_final)*(1+math.cos((epoch*3.14)/epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Training stage
ne=[]
lo_ne=[]
test_ne=[]
opt.nEpochs=50
bestLoss=10
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-5)

for epoch in range(1, opt.nEpochs + 1):
    lr = adjust_learning_rate(optimizer, epoch,1e-3,6e-4)
    print(lr)
    epoch_loss = 0
    epoch_loss1 = 0
    epoch_loss2 = 0
    model.train()
    nm=[]
    lo=[]
    for iteration, (pilot,batch) in enumerate(train_loader, 1):
        
        a,b,c,d=batch.size()
        LR = Variable(pilot).to(device)
        HR_8_target = Variable(batch).to(device)
        optimizer.zero_grad()
        HR_8 = model(LR)
        
        loss = CharbonnierLoss(HR_8.float(), HR_8_target.float())
        loss.backward()
        optimizer.step()   
           
        nm.append(loss.item())
    
        if iteration%200==0:
            tr_nmse = NMSE(HR_8_target.cpu().detach().numpy(), HR_8.cpu().detach().numpy())
            print("===> Epoch[{}]({}/{}): loss: {:.4f} nmse: {:.4f}".format(
            epoch, iteration, len(train_loader),loss.item(),10*np.log10(tr_nmse)))
     
    nmse_n=np.sum(nm)/len(nm)
    print("===> Epoch {}, Avg. Loss: {:.4f}".format(epoch,nmse_n))
    
    ne.append(nmse_n)
    
    # Vadilation stage
    nm1=[]
    nl1=[]
    model.eval()
    with torch.no_grad():
        for iteration, (pilot,batch) in enumerate(val_loader, 1):
                
            a,b,c,d=batch.size()
        
            if cuda:
                LR = pilot.cuda()
                HR_8_target = batch.cuda()

            HR_8 = model(LR)

            loss = CharbonnierLoss(HR_8.float(), HR_8_target.float())
            nmsei=np.zeros([a, 1])
            
            te_nmse = NMSE(batch.cpu().detach().numpy(), HR_8.cpu().detach().numpy())
            nm1.append(te_nmse)
            nl1.append(loss.item())
            
            
        nmse=np.sum(nm1)/len(nm1)
        nmse_db=10*np.log10(nmse)
        nl=np.sum(nl1)/len(nl1)
        print("===> loss: {:.4f}".format(nl))
        print("===> test-NMSE: {:.4f} dB".format(nmse_db))
        test_ne.append(nmse_db)
     
    # Model saving
    # Two optional saving methods
    if (epoch+1) % 50 == 0:
        net_g_model_out_path = "checkpoint/UMLP_model_epoch_{}.pth".format(epoch)
        torch.save(model, net_g_model_out_path)
    # Depends on Vadilation dataset performance
    if nl < bestLoss:
        torch.save(model, "checkpoint/UMLP_model_P64.pth")
        print("Model saved")
        bestLoss = nl
 
# Feature visualization
# plt.imshow(LR[5,0,:,:].cpu().detach().numpy())
# plt.colorbar()
# plt.figure()
# plt.imshow(HR_8_target[5,0,:,:].cpu().detach().numpy())
# plt.colorbar()