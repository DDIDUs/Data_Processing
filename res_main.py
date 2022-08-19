import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from res_model import *
import os
import torchvision.models as models

from pre_data import *
import sys

learning_rate = 0.1

def lr_scheduler(optimizer, early):
    lr = learning_rate
    if early.early_stop % 6 == 0:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

 
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def train(is_train=True, data = None, r = "a.txt", sp = 2):
    device = torch.device('cuda:0')
    
    batch_size = 200
    
    train_loader = None
    vaild_loader = None
    test_loader = None
    
    if data == "mnist":
        train_loader, vaild_loader, test_loader = Load_MNIST(sp, bs=batch_size)
    else:
        train_loader, vaild_loader, test_loader = Load_Cifar10(sp, bs=batch_size)
    
    if is_train == True:
        model = ResNet50(data)
        #model.load_state_dict(torch.load("checkpoint.pt"))
        model.apply(init_weights)
        model = model.to(device)
        num_epoch = 100
        model_name = 'model.pth'
        f = open(r, "w")
        early = EarlyStopping(patience=30)

        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=150, T_mult=1)
        train_loss = 0
        valid_loss = 0
        correct = 0
        total_cnt = 0
        best_acc = 0
        for epoch in range(num_epoch):
            print(f"====== { epoch+1} epoch of { num_epoch } ======")
            model.train()
            #lr_scheduler(optimizer, early)
            train_loss = 0
            valid_loss = 0
            correct = 0
            total_cnt = 0  
            batch = []
            if sp == 2:
                train = train_loader[int(epoch/10)]
            else:
                train = train_loader
            for step, batch in enumerate(train):
                if sp == 3:
                    index = torch.randperm(batch[0].size(0))
                    image_ = torch.index_select(batch[0], dim=0, index=index)
                    label_ = torch.index_select(batch[1], dim=0, index=index)
                    batch[0] = image_.to(device)
                    batch[1] = label_.to(device)
                else:
                    batch[0] = batch[0].to(device)
                    batch[1] = batch[1].to(device)
                
                optimizer.zero_grad()
                logits = model(batch[0])
                loss = loss_fn(logits, batch[1])
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predict = logits.max(1)

                total_cnt += batch[1].size(0)
                correct +=  predict.eq(batch[1]).sum().item()

                if step % 100 == 0 and step != 0:
                    print(f"\n====== { step } Step of { len(train_loader) } ======")
                    print(f"Train Acc : { correct / total_cnt }")
                    print(f"Train Loss : { loss.item() / batch[1].size(0) }")

            correct = 0
            total_cnt = 0
            
            with torch.no_grad():
                model.eval()
                for step, batch in enumerate(vaild_loader):
                    batch[0] = batch[0].to(device)
                    batch[1] = batch[1].to(device)
                    total_cnt += batch[1].size(0)
                    logits = model(batch[0])
                    valid_loss += loss_fn(logits, batch[1])
                    _, predict = logits.max(1)
                    correct += predict.eq(batch[1]).sum().item()
                valid_acc = correct / total_cnt
                f.write(f"\nValid Acc : { valid_acc }")    
                print(f"Valid Loss : { valid_loss / total_cnt }")
                if(valid_acc > best_acc):
                    best_acc = valid_acc
                    torch.save(model, model_name)
                    print("Model Saved!")
                    
            early(val_loss= (valid_loss / total_cnt), model=model)
            scheduler.step()
            
            if early.early_stop:
                print("stop")
                break
        f.close()
    else:
        model = torch.load("model34.pth").to(device)
        model.eval()
        
        correct = 0
        total_cnt = 0
        
        for step, batch in enumerate(test_loader):
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            total_cnt += batch[1].size(0)
            logits = model(batch[0])
            _, predict = logits.max(1)
            correct += predict.eq(batch[1]).sum().item()
        print(correct, total_cnt)
        valid_acc = correct / total_cnt
        print(f"\nTest Acc : { valid_acc }")
    
if __name__ == '__main__':
    train(is_train=True, data = sys.argv[1], r= sys.argv[2], sp= int(sys.argv[3]))