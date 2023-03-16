# -*- coding: utf-8 -*-
"""
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
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# loading Testing dataset
import h5py
path="inHmix_73_32_512_test4users_64pilot.mat"

dataset='20_dB1users_32pilot'
with h5py.File(path, 'r') as file:
    test_h = np.transpose(np.array(file['Hd']))
    test_h = test_h.transpose([0,3,1,2])

with h5py.File(path, 'r') as file:
    test_y = np.transpose(np.array(file['Yd']))
    test_y = test_y.transpose([0,3,1,2])

X_test=torch.Tensor(test_y)
y_test=torch.Tensor(test_h)

test_set = Data.TensorDataset(X_test, y_test) 

batchsize = 10

test_loader = torch.utils.data.DataLoader(test_set, batch_size= batchsize, shuffle=False)

# loading model
model_path1 = "checkpoint/UMLP_model_P64.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path1).to(device)
model.to(device)
model.eval()

# predicting stage
nmse1_snr=[]
test_nmse=[]
test = []

for i,batch in enumerate(test_loader, 1):
    input, target = batch[0].to(device), batch[1].to(device)
    
    prediction3 = model(input)
    
    tenmse = NMSE(target.cpu().detach().numpy(), prediction3.cpu().detach().numpy())
    test.append(tenmse)
    test_nmse.append(tenmse)
    if i%40==0:
        nmse1=np.mean(test_nmse)
        nmse1_snr.append(nmse1)
        test_nmse=[]

# plot figure SNR v.s. NMSE
nmse1_db=10*np.log10(nmse1_snr)
snrs = np.linspace(-10,20,7)

plt.plot(snrs, nmse1_db)
plt.grid(True) 
plt.xlabel('SNR/dB')
plt.ylabel('NMSE/dB')
plt.show()