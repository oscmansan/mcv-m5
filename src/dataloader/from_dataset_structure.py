import os

import numpy as np
import torch
from easydict import EasyDict
from sklearn.preprocessing import LabelEncoder

from .dataloader import DataLoader


class FromDatasetStructure(DataLoader):

    def __init__(self, cf: EasyDict, path: str, resize=None, preprocess=None, transform=None, valid=False,
                 predict=False):
        super().__init__()
        self.cf = cf
        self.resize = resize
        self.transform = transform
        self.preprocess = preprocess
        self.path = path
        self.indexes = None
        self.image_names = []
        self.gt = []
        self.predict = predict

        print('Reading images from {}'.format(path))

        cf.labels = os.listdir(path)
        for label in cf.labels:
            for f in os.listdir(os.path.join(path, label)):
                self.image_names.append(os.path.abspath(os.path.join(path, label, f)))
                self.gt.append(label)

        self.num_images = len(self.gt)

        le = LabelEncoder()
        le.fit(cf.labels)
        cf.map_labels = dict(zip(cf.labels, le.transform(cf.labels)))
        self.gt = le.transform(self.gt)

        if len(self.gt) != len(self.image_names):
            raise ValueError('number of images != number of labels')

        print('Found {} images belonging to {} classes'.format(len(self.image_names), len(cf.labels)))

        if len(self.image_names) < self.num_images or self.num_images == -1:
            self.num_images = len(self.image_names)
        self.img_indexes = np.arange(len(self.image_names))
        self.update_indexes(valid=valid)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_path = self.image_names[self.indexes[idx]]
        img = np.asarray(self.load_image(img_path, self.resize, self.cf.grayscale))
        gt = [self.gt[self.indexes[idx]]]
        if self.transform is not None:
            img, _ = self.transform(img, None)
        if self.preprocess is not None:
            img = self.preprocess(img)
        gt = torch.from_numpy(np.array(gt, dtype=np.int32)).long()
        if self.predict:
            return img, img_path.split("/")[-1], img.shape
        else:
            return img, gt

    def update_indexes(self, num_images=None, valid=False):
        if self.cf.shuffle and not valid:
            np.random.shuffle(self.img_indexes)
        if num_images is not None:
            if len(self.image_names) < self.num_images or num_images == -1:
                self.num_images = len(self.image_names)
            else:
                self.num_images = num_images
        self.indexes = self.img_indexes[:self.num_images]
