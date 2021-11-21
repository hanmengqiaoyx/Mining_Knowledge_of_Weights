import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from model import ResNet34
from utils import progress_bar
import numpy
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch Radiomics Training')
parser.add_argument('--lr0', default=1e-1, type=float, help='learning rate0')
parser.add_argument('--lr1', default=1e-1, type=float, help='learning rate1')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--max_epoch', default=1000, type=int, help='max epoch')
parser.add_argument('--flod', '-f', default=1, type=int, help='test flod')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()
gpu = "0"           # which GPU to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tr_loss0 = 1000
tr_loss1 = 1000
best_acc0 = 0        # best test accuracy
best_acc1 = 0
start_epoch = 0     # start from epoch 0 or last checkpoint epoch
flod = args.flod
work = True
weight_decay = 0.000


# Data
def data_prepare():
    trainloader = DataLoader(torch.load('../MiniImageNet/MiniImageNet_Train.t7'), batch_size=64, shuffle=True)
    testloader = DataLoader(torch.load('../hanmq/MiniImageNet/MiniImageNet_Test.t7'), batch_size=64, shuffle=True)
    return trainloader, testloader


# Model
def model_prepare(work):
    print('==> Building model..')
    global best_acc0
    global start_epoch
    if work == True:
        net = ResNet34()
    net = net.to(device)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    optimizer0 = optim.SGD(net.parameters(), lr=args.lr0)
    optimizer1 = optim.SGD(net.parameters(), lr=args.lr1)
    # torch.optim.lr_scheduler.StepLR(optimizer, 60, gamma=0.1, last_epoch=-1)
    # torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 100, 200, 200, 300], gamma=0.1, last_epoch=-1)
    scheduler0 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer0, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    criterion0 = nn.CrossEntropyLoss()
    criterion1 = nn.CrossEntropyLoss()
    # checkpoint = torch.load('./checkpoint/flod.t7')
    # net.load_state_dict(checkpoint['net'])
    return net, optimizer0, optimizer1, scheduler0, criterion0, scheduler1, criterion1


def train(epoch, dataloader, net, optimizer0, criterion0, optimizer1, criterion1, vali=True):
    """Train the network"""
    print('\nEpoch: %d' % epoch)
    global tr_loss0, tr_loss1
    net.train()
    num_id0 = 1e-10
    num_id1 = 1e-10
    train_loss0 = 0
    correct0 = 0
    total0 = 1e-10
    train_loss1 = 0
    correct1 = 0
    total1 = 1e-10
    gate = 0
    for batch_id, (inputs, targets) in enumerate(dataloader):
        # if batch_id < (12800 / args.batch_size):
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, gate_prt = net(inputs, epoch)
        if epoch % 2 == 0:
            gate = gate_prt
            num_id0 += 1
            loss0 = criterion0(outputs, targets.long())
            loss0.backward()
            optimizer0.step()

            train_loss0 += loss0.item()
            _, predicted0 = outputs.max(1)
            total0 += targets.size(0)
            correct0 += predicted0.eq(targets).sum().item()
            progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                         % (train_loss0 / num_id0, 100. * correct0 / total0, correct0, total0))
        elif epoch % 2 != 0:
            gate = gate_prt
            num_id1 += 1
            loss1 = criterion1(outputs, targets.long())
            loss1.backward()
            optimizer1.step()

            train_loss1 += loss1.item()
            _, predicted1 = outputs.max(1)
            total1 += targets.size(0)
            correct1 += predicted1.eq(targets).sum().item()
            progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f (%d/%d)'
                         % (train_loss1 / num_id1, 100. * correct1 / total1, correct1, total1))
        # else:
        #     print('End of the train')
        #     break
    if vali is True:
        if epoch % 2 == 0:
            tr_loss0 = train_loss0 / num_id0
        elif epoch % 2 != 0:
            tr_loss1 = train_loss1 / num_id1
    return train_loss0 / num_id0, 100. * correct0 / total0, train_loss1 / num_id1, 100. * correct1 / total1, \
           gate


def test(epoch, dataloader, net, criterion0, vali=True):
    """Validation and the test."""
    global best_acc0
    net.eval()
    num_id = 0
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_id, (inputs, targets) in enumerate(dataloader):
            # if batch_id < (2560 / args.batch_size):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, gate_prt = net(inputs, epoch)
            num_id += 1
            loss0 = criterion0(outputs, targets.long())

            test_loss += loss0.item()
            _, predicted = outputs.max(1)  # judge max elements in predicted`s Row(1:Row     0:Column)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()  # judge how many elements same in predicted and targets
            progress_bar(batch_id, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / num_id, 100. * correct / total, correct, total))
            # else:
            #     print('End of the test')
            # break
    return test_loss / num_id, 100. * correct / total


if __name__ == '__main__':
    epoch0 = 0
    epoch1 = 0
    trainloader, testloader = data_prepare()
    net, optimizer0, optimizer1, scheduler0, criterion0, scheduler1, criterion1 = model_prepare(work)
    train_loss_list0, train_acc_list0, train_loss_list1, train_acc_list1 = [], [], [], []
    for epoch in range(start_epoch, start_epoch+args.max_epoch):
        train_loss0, train_acc0, train_loss1, train_acc1, gate = train(epoch, trainloader, net, optimizer0, criterion0, optimizer1, criterion1)
        scheduler0.step(tr_loss0)
        scheduler1.step(tr_loss1)
        lr0 = optimizer0.param_groups[0]['lr']
        lr1 = optimizer1.param_groups[0]['lr']
        train_loss_array0 = numpy.array(train_loss_list0)
        train_acc_array0 = numpy.array(train_acc_list0)
        train_loss_array1 = numpy.array(train_loss_list1)
        train_acc_array1 = numpy.array(train_acc_list1)
        if lr0 < 5e-4 and lr1 < 5e-4:
            test_loss, test_acc = test(epoch, testloader, net, criterion0)
            print('Saving:')
            state1 = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('acc'):
                os.mkdir('acc')
            torch.save(state1, './acc/flod''.t7')
            acc = open('ResNet34.txt', 'w')
            acc.write(str(test_acc))
            acc.close()
            print('Saving:')
            plt.figure(1)
            plt.subplot(2, 2, 1)
            plt.xlabel('epoch')
            plt.ylabel('train loss')
            plt.plot([i for i in range(epoch0)], train_loss_array0, '-')
            plt.subplot(2, 2, 2)
            plt.xlabel('epoch')
            plt.ylabel('train acc')
            plt.plot([i for i in range(epoch0)], train_acc_array0, '-')
            plt.savefig("ResNet34_First_CE.jpg")
            plt.figure(2)
            plt.subplot(2, 2, 1)
            plt.xlabel('epoch')
            plt.ylabel('train loss')
            plt.plot([i for i in range(epoch1)], train_loss_array1, '-')
            plt.subplot(2, 2, 2)
            plt.xlabel('epoch')
            plt.ylabel('train acc')
            plt.plot([i for i in range(epoch1)], train_acc_array1, '-')
            plt.savefig("ResNet34_Second_CE.jpg")
            plt.show()
            print('OVER')
            break
        else:
            pass