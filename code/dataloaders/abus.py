import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
import SimpleITK as sitk
import random 

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
from torchvision import transforms


class ABUS(Dataset):
    """ ABUS dataset """
    def __init__(self, base_dir=None, split='train', fold='1', num=None, use_dismap=False, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.split = split
        self.use_dismap = use_dismap 
        if split=='train':
            with open(self._base_dir+'/../abus_train.list.'+fold, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir+'/../abus_test.list.'+fold, 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image = self.load_img('image/'+image_name, is_normalize=True)
        label = self.load_img('label/'+image_name)
        label[label!=0] = 1
        if self.use_dismap:
            dis_map = self.load_img('dis_map/'+image_name[:-3]+'nii.gz')
            sample = {'image': image, 'label': label, 'dis_map': dis_map}
        else:
            sample = {'image': image, 'label': label}

        #print('image.shape', image.shape)
        if self.transform:
            sample = self.transform(sample)
        
        if self.split == 'test':
            sample['filename'] = image_name

        return sample
    
    def load_img(self, image_name, is_normalize=False):
        filename = os.path.join(self._base_dir, image_name)
        itk_img = sitk.ReadImage(filename)
        image = sitk.GetArrayFromImage(itk_img)
        #image = np.transpose(image, (1,2,0))
        image = image.astype(np.float32)
        #print('image.shape: ', image.shape)

        if is_normalize:
            #print('image.max ', image.max())
            #print('image.min', image.min())
            image = image / image.max() 
            image = image / 0.5 - 0.5

        return image

class CenterCrop(object):
    def __init__(self, output_size, use_dismap=False):
        self.output_size = output_size
        self.use_dismap = use_dismap

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.use_dismap:
            dis_map = sample['dis_map']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.use_dismap:
                dis_map = np.pad(dis_map, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=1.)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        sample['image'], sample['label'] = image, label

        if self.use_dismap:
            dis_map = dis_map[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            sample['dis_map'] = dis_map

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, use_dismap=False):
        self.output_size = output_size
        self.use_dismap = use_dismap

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.use_dismap:
            dis_map = sample['dis_map']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.use_dismap:
                dis_map = np.pad(dis_map, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=1.)


        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.use_dismap:
            dis_map = dis_map[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            sample['dis_map'] = dis_map
        sample['image'], sample['label'] = image, label

        return sample 


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, probability=0.6, use_dismap=False):
        self.probability = probability
        self.use_dismap = use_dismap

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.use_dismap:
            dis_map = sample['dis_map']

        if round(np.random.uniform(0,1),1) <= self.probability:
            k = random.choices([2,4],k=1)
            k = k[0]
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            if self.use_dismap:
                dis_map = np.rot90(dis_map, k)

            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            if self.use_dismap:
                dis_map = np.flip(dis_map, axis=axis).copy()

            sample['image'], sample['label'] = image, label
            if self.use_dismap:
                sample['dis_map'] = dis_map

        return sample 


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, use_dismap=False):
        self.use_dismap = use_dismap

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample['image'] = torch.from_numpy(image)
        sample['label'] = torch.from_numpy(label).long()
        if self.use_dismap:
            dis_map = sample['dis_map']
            dis_map = np.expand_dims(dis_map, 0)
            #print('dis_map.shape: ', dis_map.shape)
            sample['dis_map'] = torch.from_numpy(dis_map.astype(np.float32))

        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)



if __name__ == "__main__":
    db_train = ABUS(base_dir='../../data/abus_data/',
                       split='train',
                       use_dismap=True,
                       transform = transforms.Compose([RandomRotFlip(use_dismap=True), 
                                                       RandomCrop((128, 64, 128), use_dismap=True), 
                                                       ToTensor()]))
    def worker_init_fn(worker_id):
        random.seed(2009+worker_id)
    trainloader = DataLoader(db_train, batch_size=2, shuffle=True,  num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    for sample in trainloader:
        image, label, dis_map = sample['image'], sample['label'], sample['dis_map']
        print('image', image.shape)
        print('label', label.shape)
        print('dis_map', dis_map.shape)
