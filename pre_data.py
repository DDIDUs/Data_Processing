from random import shuffle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as model
import random
import numpy as np


def Load_Cifar10(sp=None, bs = 200):
    
    label1 = [0,1,2,3,4,5,6,7,8,9]
    
    train_data_len = []
    train_label = []
    train_loader = []
    min_len = 0
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='./data2', train=True, download=True, transform=transform_train)
    train_dataset, vaild_dataset = torch.utils.data.random_split(dataset, [45000, 5000])
    test_dataset = torchvision.datasets.CIFAR10(root='./data2', train=False, download=True, transform=transform_test)
    
    if sp != 1:
        for i in label1:
            t = []
            for j in train_dataset:
                if j[1] == i:
                    t.append(j)
            train_data_len.append(len(t))
            train_label.append(t)

        min_len = min(train_data_len)
    
    if sp == 1:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8)
    elif sp == 2:
        for i in train_label:
            t = i[:min_len]
            train_loader.append(torch.utils.data.DataLoader(tuple(t), batch_size=bs, shuffle=False, num_workers=8))
    else:
        #train_loader = torch.utils.data.DataLoader(tuple(sorted(train_dataset, key=lambda x: x[1])), batch_size=256, shuffle=False, num_workers=8)
        temp = []
        for i in range(min_len):
            w = []
            for t in train_label:
                w.append(t.pop())
            temp+=w
        train_loader = torch.utils.data.DataLoader(tuple(temp), batch_size=bs, shuffle=False, num_workers=8)
        
    vaild_loader = torch.utils.data.DataLoader(vaild_dataset, batch_size=bs, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=8)
    return train_loader, vaild_loader, test_loader

def Load_MNIST(sp = None, bs = 200):
    
    label1 = [0,1,2,3,4,5,6,7,8,9]
    
    train_data_len = []
    train_label = []
    train_loader = []
    min_len = 0
    
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_dataset, vaild_dataset = torch.utils.data.random_split(dataset, [50000, 10000])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    if sp != 1:
        for i in label1:
            t = []
            for j in train_dataset:
                if j[1] == i:
                    t.append(j)
            train_data_len.append(len(t))
            train_label.append(t)

        min_len = min(train_data_len)
    
    if sp == 1:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8)
    elif sp == 2:
        for i in train_label:
            t = i[:min_len]
            train_loader.append(torch.utils.data.DataLoader(tuple(t), batch_size=bs, shuffle=False, num_workers=8))
    else:
        #train_loader = torch.utils.data.DataLoader(tuple(sorted(train_dataset, key=lambda x: x[1])), batch_size=256, shuffle=False, num_workers=8)
        temp = []
        for i in range(min_len):
            w = []
            for t in train_label:
                w.append(t.pop())
            temp+=w
        train_loader = torch.utils.data.DataLoader(tuple(temp), batch_size=bs, shuffle=False, num_workers=8)
        
    vaild_loader = torch.utils.data.DataLoader(vaild_dataset, batch_size=bs, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=True, num_workers=8)
    return train_loader, vaild_loader, test_loader