import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from vgg_model import *
from pre_data import *
import sys

def lr_scheduler(optimizer, early):
    lr = learning_rate
    if early.counter > 6:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

learning_rate = 0.01
epoch = 100

def train(is_train=True, data = None, kdkd = None, r = None, sp = None):
    device = torch.device("cuda:0")
    f = open(r, "w")
    batch_size = 200
    if data == "mnist":
        train_loader, valid_loader, test_loader = Load_MNIST(sp,bs=batch_size)
    else:
        train_loader, valid_loader, test_loader = Load_Cifar10(sp,bs=batch_size)
    
    if is_train == True:
        early = EarlyStopping(patience=30)
        #VGG 클래스를 인스턴스화
        if data == "mnist":
            model = VGG("VGG16m", data)
        else:
            model = VGG("VGG16", data)
        model = model.to(device)

        # 손실함수 및 최적화함수 설정
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        loss_arr = []

        for i in range(epoch):
            model.train()
            print("=====", i, "Step of ", epoch, "=====")
            
            if sp == 2:
                train = train_loader[int(i/10)]
            else:
                train = train_loader             
            
            for j, batch in enumerate(train):
                if sp == 3:
                    index = torch.randperm(batch[0].size(0))
                    image_ = torch.index_select(batch[0], dim=0, index=index)
                    label_ = torch.index_select(batch[1], dim=0, index=index)
                    batch[0] = image_.to(device)
                    batch[1] = label_.to(device)
                x, y_ = batch[0].to(device), batch[1].to(device)
                #lr_scheduler(optimizer, early)
                optimizer.zero_grad()
                output = model(x)
                loss = loss_func(output,y_)
                loss.backward()
                optimizer.step()

            if i % 10 ==0:
                loss_arr.append(loss.cpu().detach().numpy())

            correct = 0
            total = 0
            valid_loss = 0
            best_acc = 0
            
            model.eval()
            with torch.no_grad():
                for image,label in valid_loader:
                    x = image.to(device)
                    y = label.to(device)
                        
                    output = model.forward(x)
                    valid_loss += loss_func(output, y)
                    _,output_index = torch.max(output,1)

                    total += label.size(0)
                    correct += (output_index == y).sum().float()
                print("loss : ", {valid_loss/total})
                t = "Accuracy of Test Data: {}%".format(100*correct/total)
                f.write(t)
                print(t)
                if correct/total > best_acc:
                    best_acc = correct/total
                    torch.save(model, kdkd)
                    print("model saved")

            early((valid_loss/total), model)
            
            if early.early_stop:
                print("stop")
                break
            scheduler.step()
        f.close()
    else:
        model = torch.load("model1-cifar10.pth").to(device)
        model.eval()
        correct = 0
        total_cnt = 0
        
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        
        for step, batch in enumerate(test_loader):
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            total_cnt += batch[1].size(0)
            logits = model(batch[0])
            _, predict = logits.max(1)
            correct += predict.eq(batch[1]).sum().item()
            c = (predict == batch[1]).squeeze()
            for i in range(len(batch[1])):
                label = batch[1][i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
        print(correct, total_cnt)
        valid_acc = correct / total_cnt
        print(f"\nTest Acc : { valid_acc }")    
        
        classes = [0,1,2,3,4,5,6,7,8,9]
        
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
            
if __name__ == '__main__':
    train(is_train=True, data=sys.argv[2], kdkd= sys.argv[1], r= sys.argv[3], sp= int(sys.argv[4]))