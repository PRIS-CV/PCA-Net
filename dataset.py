import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def config(data):
    if data == 'bird':
        train_root = '/data/zhangt/Birds/train'
        test_root = '/data/zhangt/Birds/test'
        train_pd = pd.read_csv("/data/zhangt/Birds/bird_train.txt", sep=" ", header=None,
                               names=['ImageName', 'label'])
        test_pd = pd.read_csv("/data/zhangt/Birds/bird_test.txt", sep=" ", header=None,
                              names=['ImageName', 'label'])
        cls_num = 200

    if data == 'car':
        train_root = '/data/zhangt/Car/train'
        test_root = '/data/zhangt/Car/test'
        train_pd = pd.read_csv("/data/zhangt/Car/car_train.txt", sep=" ", header=None,
                               names=['ImageName', 'label'])
        test_pd = pd.read_csv("/data/zhangt/Car/car_test.txt", sep=" ", header=None,
                              names=['ImageName', 'label'])
        cls_num = 196

    if data == 'aircraft':
        train_root = '/data/zhangt/Aircraft/train'
        test_root = '/data/zhangt/Aircraft/test'
        train_pd = pd.read_csv("/data/zhangt/Aircraft/aircraft_train.txt", sep=" ", header=None,
                               names=['ImageName', 'label'])
        test_pd = pd.read_csv("/data/zhangt/Aircraft/aircraft_test.txt", sep=" ", header=None,
                              names=['ImageName', 'label'])
        cls_num = 100

    return train_root, test_root, train_pd, test_pd, cls_num


class Dataset(Dataset):
    def __init__(self, root_dir, pd_file, train=False, transform=None, num_positive=1):
        self.root_dir = root_dir
        self.pd_file = pd_file
        self.image_names = pd_file['ImageName'].tolist()
        self.labels = pd_file['label'].tolist()

        self.train = train
        self.transform = transform

        self.num_positive = num_positive

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_dir, self.image_names[item])
        image = self.pil_loader(img_path)
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)
        if self.train:
            positive_image = self.fetch_positive(self.num_positive, label, self.image_names[item])
            return image, positive_image, label
        return image, label

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def fetch_positive(self, num_positive, label, path):
        other_img_info = self.pd_file[(self.pd_file.label == label) & (self.pd_file.ImageName != path)]
        other_img_info = other_img_info.sample(min(num_positive, len(other_img_info))).to_dict('records')
        other_img_path = [os.path.join(self.root_dir, e['ImageName']) for e in other_img_info]
        other_img = [self.pil_loader(img) for img in other_img_path]
        positive_img = [self.transform(img) for img in other_img]
        return positive_img


def collate_fn(batch):
    imgs = []
    positive_imgs = []
    labels = []
    for sample in batch:
        imgs.append(sample[0])
        positive_imgs.extend(sample[1])
        labels.append(sample[2])
    return torch.stack(imgs, 0), torch.stack(positive_imgs, 0), labels
