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
from MoCo import MoCo

cpu = torch.device("cpu")
gpu = torch.device('cuda')

class Net(nn.Module):
    def __init__(self, encoder):
        super(Net, self).__init__()
        self.model = encoder
    
    def forward(self, img):
        out = self.model(img)
        out = nn.functional.normalize(out, dim=1)
        out = out.to(cpu).detach().numpy()
        return out.reshape(out.shape[1])
        
        
class Evaluate():
    def __init__(self, dataPath, model):
        self.model = model
        self.path = dataPath
        self.dir = os.listdir(dataPath)
        self.files = []
        self.embed = {}
        
    def generateEmbeddings(self):
        for f in self.dir:
            if f[-4:]!='.png':
                continue
            self.files.append(f)
            img = cv2.imread(self.path + f)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(150,140),interpolation=cv2.INTER_CUBIC)
            img = torch.from_numpy(img.reshape((1,3,img.shape[0],img.shape[1])))
            q = self.model(img.to(gpu, dtype=torch.float32))
            self.embed[f] = q
    def evaluate(self, n): #recall at n
        pos = 0
        total = 0
        rank = np.zeros(len(self.files))
        for it, i in enumerate(self.files):
            scores = {}
            for j in self.files:
                if i == j:
                    continue
                else:
                    v1 = self.embed[i]
                    v2 = self.embed[j]
                    scores[j] = -np.dot(v1, v2.T)
            ans = sorted(scores.items(), key=lambda x:x[1])
            for a in range(len(ans)):
                if ans[a][0].split('_')[0] == i.split('_')[0]:
                    rank[it] = a
                    if a<n:
                        pos += 1
            total += 1
        print("average recall @ %d : %.3f"%(n, pos/total))
        print("average rank : %.3f"%(np.average(rank)))
        
    def test(self):
        self.generateEmbeddings()
        self.evaluate(1)
        self.evaluate(5)
        self.evaluate(10)
        
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='1', type=str, required=False, help='model number')
    parser.add_argument('--folder', default='./test/', type=str, required=False, help='test image folder')
    args = parser.parse_args()
    print(args.model)
    modelPath = "./models/MoCo_"+args.model+".pt"
    model = torch.load(modelPath)
    model.eval()
    model = Net(model.encoder_q)

    myEval = Evaluate(args.folder, model)
    myEval.test()

    
