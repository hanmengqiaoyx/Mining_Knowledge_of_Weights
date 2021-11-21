import torch
from torchvision.transforms import *
from torchmeta.datasets.miniimagenet import *


class ReadMiniImageNetA_train(Dataset):
    def __init__(self, split='train', transform=None):
        self._dataset = MiniImagenetClassDataset('../Datasets/miniimagenet/', meta_split=split)
        self._features = self._dataset.data
        self._labels = self._dataset.labels
        self.transform = transform
        self.flatten()

    def flatten(self):
        datalist = []
        labellist = []
        for i in self._labels:
            for j in self._features[i]:
                datalist.append(j)
                labellist.append(i)

        self.features = datalist
        self._named_labels = labellist
        labeldict = {}
        l = 0
        for i in self._named_labels:
            if i not in labeldict.keys():
                labeldict[i] = l
                l += 1
        self.labels = list(map(lambda x: labeldict[x], self._named_labels))
        return len(self.features) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    @property
    def class_nums(self):
        return len(self._labels)

    def __getitem__(self, index):
        f = self.features[index]
        l = self.labels[index]
        if self.transform is not None:
            f = self.transform(f)
        else:
            t = Compose([ToPILImage(), Resize(84), ToTensor()])
            f = t(f)

        return (f, l)


class ReadMiniImageNetA_vaild(Dataset):
    def __init__(self, split='val', transform=None):
        self._dataset = MiniImagenetClassDataset('../Datasets/miniimagenet/', meta_split=split)
        self._features = self._dataset.data
        self._labels = self._dataset.labels
        self.transform = transform
        self.flatten()

    def flatten(self):
        datalist = []
        labellist = []
        for i in self._labels:
            for j in self._features[i]:
                datalist.append(j)
                labellist.append(i)

        self.features = datalist
        self._named_labels = labellist
        labeldict = {}
        l = 0
        for i in self._named_labels:
            if i not in labeldict.keys():
                labeldict[i] = l
                l += 1
        self.labels = list(map(lambda x: labeldict[x], self._named_labels))
        return len(self.features) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    @property
    def class_nums(self):
        return len(self._labels)

    def __getitem__(self, index):
        f = self.features[index]
        l = self.labels[index]
        if self.transform is not None:
            f = self.transform(f)
        else:
            t = Compose([ToPILImage(), Resize(84), ToTensor()])
            f = t(f)

        return (f, l)


class ReadMiniImageNetA_test(Dataset):
    def __init__(self, split='test', transform=None):
        self._dataset = MiniImagenetClassDataset('../Datasets/miniimagenet/', meta_split=split)
        self._features = self._dataset.data
        self._labels = self._dataset.labels
        self.transform = transform
        self.flatten()

    def flatten(self):
        datalist = []
        labellist = []
        for i in self._labels:
            for j in self._features[i]:
                datalist.append(j)
                labellist.append(i)

        self.features = datalist
        self._named_labels = labellist
        labeldict = {}
        l = 0
        for i in self._named_labels:
            if i not in labeldict.keys():
                labeldict[i] = l
                l += 1
        self.labels = list(map(lambda x: labeldict[x], self._named_labels))
        return len(self.features) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    @property
    def class_nums(self):
        return len(self._labels)

    def __getitem__(self, index):
        f = self.features[index]
        l = self.labels[index]
        if self.transform is not None:
            f = self.transform(f)
        else:
            t = Compose([ToPILImage(), Resize(84), ToTensor()])
            f = t(f)

        return (f, l)