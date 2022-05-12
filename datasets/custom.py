import os.path as osp
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import transform as T
from datasets.randaugment import RandomAugment
from datasets.sampler import RandomSampler, BatchSampler

from PIL import Image as Image
from myscripts.create_datasets import _decode_png

class TwoCropsTransform:
    """Take 2 random augmentations of one image."""

    def __init__(self,trans_weak,trans_strong):       
        self.trans_weak = trans_weak
        self.trans_strong = trans_strong
    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong(x)
        return [x1, x2]
    
class ThreeCropsTransform:
    """Take 3 random augmentations of one image."""

    def __init__(self,trans_weak,trans_strong0,trans_strong1):       
        self.trans_weak = trans_weak
        self.trans_strong0 = trans_strong0
        self.trans_strong1 = trans_strong1
    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong0(x)
        x3 = self.trans_strong1(x)
        return [x1, x2, x3]

def load_data_train(dspth='/opt/chenhaoqing/data/redtheme/batched_data'):
    datalist_x = [
        osp.join(dspth, 'train_batch_0')
    ]
    datalist_u = [
        osp.join(dspth, 'unlabel_batch_0')
    ]
    data_x, label_x, data_u, label_u = [], [], [], []
    for data_batch in datalist_x:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr)
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data_x.append([_decode_png(x) for x in entry['data']])        
            label_x.append(lbs)
    data_x = np.concatenate(data_x, axis=0)
    label_x = np.concatenate(label_x, axis=0)
    data_x = [
        el.reshape(3, 224, 224).transpose(1, 2, 0)
        for el in data_x
    ]
    
    for data_batch in datalist_u:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr)
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data_u.append([_decode_png(x) for x in entry['data']])        
            label_u.append(lbs)
    data_u = np.concatenate(data_u, axis=0)
    label_u = np.concatenate(label_u, axis=0)
    data_u = [
        el.reshape(3, 224, 224).transpose(1, 2, 0)
        for el in data_u
    ]
    return data_x, label_x, data_u, label_u


def load_data_val(dspth='/opt/chenhaoqing/data/redtheme/batched_data'):
    datalist = [
        osp.join(dspth, 'val_batch_0')
    ]
    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr)
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append([_decode_png(x) for x in entry['data']])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data = [
        el.reshape(3, 224, 224).transpose(1, 2, 0)
        for el in data
    ]
    return data, labels


def load_data_fp(dspth='/opt/chenhaoqing/data/redtheme/batched_data'):
    datalist = [
        osp.join(dspth, 'fp_batch_0')
    ]
    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr)
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append([_decode_png(x) for x in entry['data']])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data = [
        el.reshape(3, 224, 224).transpose(1, 2, 0)
        for el in data
    ]
    return data, labels


def load_data_recall(dspth='/opt/chenhaoqing/data/redtheme/batched_data'):
    datalist = [
        osp.join(dspth, 'recall_batch_0')
    ]
    data, labels = [], []
    for data_batch in datalist:
        with open(data_batch, 'rb') as fr:
            entry = pickle.load(fr)
            lbs = entry['labels'] if 'labels' in entry.keys() else entry['fine_labels']
            data.append([_decode_png(x) for x in entry['data']])
            labels.append(lbs)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    data = [
        el.reshape(3, 224, 224).transpose(1, 2, 0)
        for el in data
    ]
    return data, labels


def compute_mean_var():
    data_x, label_x, data_u, label_u = load_data_train()
    data = data_x + data_u
    data = np.concatenate([el[None, ...] for el in data], axis=0)

    mean, var = [], []
    for i in range(3):
        channel = (data[:, :, :, i].ravel() / 127.5) - 1
        #  channel = (data[:, :, :, i].ravel() / 255)
        mean.append(np.mean(channel))
        var.append(np.std(channel))

    print('mean: ', mean)
    print('var: ', var)



class Custom(Dataset):
    def __init__(self, dataset, data, labels, mode):
        super(Custom, self).__init__()
        self.data, self.labels = data, labels
        self.mode = mode
        assert len(self.data) == len(self.labels)
        crop_size = 224
        if dataset == 'redtheme':
            # compute_mean_var()
            mean, std = (-0.0141, -0.0276, -0.0741), (0.5770, 0.5540, 0.5687)
    
        trans_weak = T.Compose([
            T.Resize((crop_size, crop_size)),
            T.PadandRandomCrop(border=4, cropsize=(crop_size, crop_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.Normalize(mean, std),
            T.ToTensor(),
        ])
        trans_strong0 = T.Compose([
            T.Resize((crop_size, crop_size)),
            T.PadandRandomCrop(border=4, cropsize=(crop_size, crop_size)),
            T.RandomHorizontalFlip(p=0.5),
            RandomAugment(2, 10),
            T.Normalize(mean, std),
            T.ToTensor(),
        ])        
        trans_strong1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),     
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),        
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])                    
        if self.mode == 'train_x':
            self.trans = trans_weak
        elif self.mode == 'train_u_comatch':
            self.trans = ThreeCropsTransform(trans_weak, trans_strong0, trans_strong1)               
        elif self.mode == 'train_u_fixmatch':
            self.trans = TwoCropsTransform(trans_weak, trans_strong0)    
        else:  
            self.trans = T.Compose([
                T.Resize((crop_size, crop_size)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        im, lb = self.data[idx], self.labels[idx]
        return self.trans(im), lb

    def __len__(self):
        leng = len(self.data)
        return leng


def get_train_loader(dataset, batch_size, mu, n_iters_per_epoch, root='/opt/chenhaoqing/data/redtheme/batched_data', method='comatch'):
    data_x, label_x, data_u, label_u = load_data_train(dspth=root)

    ds_x = Custom(
        dataset=dataset,
        data=data_x,
        labels=label_x,
        mode='train_x'
    )  # return an iter of num_samples length (all indices of samples)
    sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
    batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
    dl_x = torch.utils.data.DataLoader(
        ds_x,
        batch_sampler=batch_sampler_x,
        num_workers=2,
        pin_memory=True
    )
    ds_u = Custom(
        dataset=dataset,
        data=data_u,
        labels=label_u,
        mode='train_u_%s'%method
    )
    sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
    batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
    dl_u = torch.utils.data.DataLoader(
        ds_u,
        batch_sampler=batch_sampler_u,
        num_workers=2,
        pin_memory=True
    )
    return dl_x, dl_u


def get_val_loader(dataset, batch_size, num_workers, pin_memory=True, root='/opt/chenhaoqing/data/redtheme/batched_data'):
    data, labels = load_data_val(dspth=root)
    ds = Custom(
        dataset=dataset,
        data=data,
        labels=labels,
        mode='test'
    )
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dl


# 业务数据测试疑似fp/召回recall
def get_test_loader(dataset, batch_size, num_workers, type, pin_memory=True, root='/opt/chenhaoqing/data/redtheme/batched_data'):
    if type == "fp":
        data, labels = load_data_fp(dspth=root)
    elif type == "recall":
        data, labels = load_data_recall(dspth=root)
    
    ds = Custom(
        dataset=dataset,
        data=data,
        labels=labels,
        mode='test'
    )
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dl


def main():
    compute_mean_var()

if __name__ == 'main':
    main()