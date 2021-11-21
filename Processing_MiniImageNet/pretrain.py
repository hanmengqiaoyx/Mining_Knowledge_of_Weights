import torch
from data import ReadMiniImageNetA_train, ReadMiniImageNetA_vaild, ReadMiniImageNetA_test
from torch.utils.data import *
import numpy as np
import torchvision.transforms as transforms


def init_dataloaders():
    trainset = ReadMiniImageNetA_train()
    trainloader = DataLoader(trainset, batch_size=100, shuffle=False)
    valset = ReadMiniImageNetA_vaild()
    valloader = DataLoader(valset, batch_size=100, shuffle=False)
    testset = ReadMiniImageNetA_test()
    testloader = DataLoader(testset, batch_size=100, shuffle=False)
    return trainloader, valloader, testloader


if __name__ == '__main__':
    trainloader, valloader, testloader = init_dataloaders()
    train_list = []
    label0_list = []
    test_list = []
    label1_list = []
    i = 5
    j = 5
    k = 5
    j_label = 64
    k_label = 80
    print('////first////:')
    for batch_id, (inputs, targets) in enumerate(trainloader):
        if batch_id == i:
            print('////Test////', batch_id)
            test_list.extend(np.array(inputs))
            label1_list.extend(np.array(targets))
            i += 6
        else:
            print('////Train////', batch_id)
            train_list.extend(np.array(inputs))
            label0_list.extend(np.array(targets))

    print('////second////:')
    for batch_id, (inputs, targets) in enumerate(valloader):
        if batch_id == j:
            print('////Test////', batch_id)
            test_list.extend(np.array(inputs))
            label1_list.extend(np.array(targets+j_label))
            j += 6
        else:
            print('////Train////', batch_id)
            train_list.extend(np.array(inputs))
            label0_list.extend(np.array(targets+j_label))

    print('////third////:')
    for batch_id, (inputs, targets) in enumerate(testloader):
        if batch_id == k:
            print('////Test////', batch_id)
            test_list.extend(np.array(inputs))
            label1_list.extend(np.array(targets+k_label))
            k += 6
        else:
            print('////Train////', batch_id)
            train_list.extend(np.array(inputs))
            label0_list.extend(np.array(targets+k_label))

    train_data = np.array(train_list)
    train_data = torch.from_numpy(train_data)
    label0 = np.array(label0_list)
    label0 = torch.from_numpy(label0)
    test_data = np.array(test_list)
    test_data = torch.from_numpy(test_data)
    label1 = np.array(label1_list)
    label1 = torch.from_numpy(label1)

    Train_Dataset = TensorDataset(train_data, label0)
    torch.save(Train_Dataset, './MiniImageNet_Train''.t7')
    Test_Dataset = TensorDataset(test_data, label1)
    torch.save(Test_Dataset, './MiniImageNet_Test''.t7')

    # dataloader = DataLoader(torch.load('../../../hanmq/MiniImageNet/MiniImageNet_Train.t7'), batch_size=500, shuffle=False)
    # for batch_id, (inputs, targets) in enumerate(dataloader):
    #     print(batch_id)
    #     print(targets)

        # image = inputs[4]
        # image = transforms.ToPILImage()(image)
        # image.show()

    print('END')