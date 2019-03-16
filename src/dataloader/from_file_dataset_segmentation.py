import numpy as np
import torch

from .dataloader import DataLoader


class FromFileDatasetSegmentation(DataLoader):

    def __init__(self, cf, image_txt, gt_txt, num_images, resize=None,
                 preprocess=None, transform=None, valid=False):
        super(FromFileDatasetSegmentation, self).__init__()

        self.cf = cf
        self.resize = resize
        self.transform = transform
        self.preprocess = preprocess
        self.num_images = num_images

        print("Reading images from: {}".format(image_txt))
        with open(image_txt) as f:
            image_names = f.readlines()
        self.image_names = [x.strip() for x in image_names]

        print("Reading annotations from: {}".format(gt_txt))
        with open(gt_txt) as f:
            gt_names = f.readlines()
        self.gt_names = [x.strip() for x in gt_names]

        if len(self.gt_names) != len(self.image_names):
            raise ValueError('number of images != number of annotation masks')

        print("Found {} images".format(len(self.image_names)))

        if len(self.image_names) < self.num_images or self.num_images == -1:
            self.num_images = len(self.image_names)
        self.img_indexes = np.arange(len(self.image_names))
        self.update_indexes(valid=valid)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_path = self.image_names[self.indexes[idx]]
        gt_path = self.gt_names[self.indexes[idx]]

        img, _ = self.load_img(img_path, self.resize, self.cf.grayscale, order=1)

        if self.cf.map_labels is not None:
            gt, _ = self.cf.map_labels[self.load_img(gt_path, self.resize, grayscale=True, order=0)]
        else:
            gt, _ = self.load_img(gt_path, self.resize, grayscale=True, order=0)

        if self.transform is not None:
            img, gt = self.transform(img, gt)
        if self.preprocess is not None:
            img = self.preprocess(img)

        gt = torch.from_numpy(np.array(gt, dtype=np.int32)).long()

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
