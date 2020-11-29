import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnext50_32x4d
from torchvision.models import resnet18
import time
from tqdm import tqdm
from dataset import dataset
import os
import cv2
import argparse
cpu = torch.device("cpu")
gpu = torch.device('cuda')

def adjust_learning_rate(optimizer, epoch):
    """Set learning rate decay every 50 epochs"""
    if epoch%50 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
class MoCo(nn.Module):
    def __init__(self, dim = 128, K = 128, m = 0.99, T = 0.07):
        super(MoCo, self).__init__()
        """ 
        dim: output dimension 
        K: dictionary size 
        m: momentum parameter 
        T: temperature parameter 
        """
        self.dim = dim
        self.K = K
        self.m = m
        self.T = T

        """ applying resnet18 or resnet50 as encoder """
#         self.encoder_q = resnext50_32x4d(pretrained=False)
#         self.encoder_k = resnext50_32x4d(pretrained=False)
        self.encoder_q = resnet18(pretrained=True)
        self.encoder_k =  resnet18(pretrained=True)
        
        """ make sure the output dimension is as our expected """
        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, self.dim))
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, self.dim))
        
        """ fixed the parameters of key encoder which is updated by momentum contrast """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    def _momentum_update_key_encoder(self):
        """ Momentum update of the key encoder """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
    def _dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        """ dequeue and enqueue. I have modified it for more general """
        if ptr + batch_size > self.K:
            rest = self.K - ptr
            self.queue[:, ptr:] = keys[0:rest].T
            self.queue[:, 0:(batch_size-rest)] = keys[rest:].T
            ptr = batch_size-rest
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        """ THIS PART IS REFERENCED FROM ORIGINAL MOCO RELEASED CODE """
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)


        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        return logits, labels

if __name__ == '__main__':
    """ parameters settings """
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=100, type=int, required=False, help='epoch')
    parser.add_argument('--folder', default='./train/', type=str, required=False, help='train image folder')
    parser.add_argument('--dim', default=128, type=int, required=False, help='output dimension')
    args = parser.parse_args()
    
    """ model definition """
    model = MoCo(args.dim)
    model.to(gpu)
    print(model)
    model.train()
    creterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4)

    """ Data Loader """
    datas = dataset(args.folder)
    loader = Data.DataLoader(dataset=datas, batch_size=64, shuffle=True)
    tqd = tqdm(range(args.epoch))
    
    x1 = range(0, args.epoch)
    y1 = []
    best_loss = 99999
    best_i = -1
    
    """ training """
    for i in tqd:
        local_loss = 0
        adjust_learning_rate(optimizer, i+1)
        
        for step, (key, query) in enumerate(loader):
            optimizer.zero_grad()
            logits, labels = model(query.to(gpu, dtype=torch.float32), key.to(gpu, dtype=torch.float32))
            loss = creterion(logits, labels)
            local_loss += loss.data
        

        if local_loss <= best_loss and i%10 == 0:
            best_i = i
            best_loss = local_loss
            torch.save(model, './models/MoCo_'+str(i)+'.pt')
            tqd.set_postfix_str("best epoch: %2d, loss: %.3f"%(best_i + 1, best_loss))
            
        y1.append(local_loss)

    """ plot the training results """
    plt.subplot(1, 1, 1)
    plt.plot(x1, y1, '-')
    plt.title('Training loss vs. epoches')
    plt.ylabel('Training loss')
    plt.savefig('loss.jpg')

