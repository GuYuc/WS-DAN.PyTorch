""" FGVC Aircraft (Aircraft) Dataset
Created: Nov 15,2019 - Yuchong Gu
Revised: Nov 15,2019 - Yuchong Gu
"""
import os
import pdb
from PIL import Image
from torch.utils.data import Dataset
from utils import get_transform

DATAPATH = '/home/guyuchong/DATA/FGVC/FGVC-Aircraft/data'
FILENAME_LENGTH = 7


class AircraftDataset(Dataset):
    """
    # Description:
        Dataset for retrieving FGVC Aircraft images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', resize=500):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize

        variants_dict = {}
        with open(os.path.join(DATAPATH, 'variants.txt'), 'r') as f:
            for idx, line in enumerate(f.readlines()):
                variants_dict[line.strip()] = idx
        self.num_classes = len(variants_dict)

        if phase == 'train':
            list_path = os.path.join(DATAPATH, 'images_variant_trainval.txt')
        else:
            list_path = os.path.join(DATAPATH, 'images_variant_test.txt')

        self.images = []
        self.labels = []
        with open(list_path, 'r') as f:
            for line in f.readlines():
                fname_and_variant = line.strip()
                self.images.append(fname_and_variant[:FILENAME_LENGTH])
                self.labels.append(variants_dict[fname_and_variant[FILENAME_LENGTH + 1:]])

        # transform
        self.transform = get_transform(self.resize, self.phase)

    def __getitem__(self, item):
        # image
        image = Image.open(os.path.join(DATAPATH, 'images', '%s.jpg' % self.images[item])).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        # return image and label
        return image, self.labels[item]  # count begin from zero

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    ds = AircraftDataset('test', 448)
    # print(len(ds))
    from utils import AverageMeter
    height_meter = AverageMeter('height')
    width_meter = AverageMeter('width')

    for i in range(len(ds)):
        image, label = ds[i]
        avgH = height_meter(image.size(1))
        avgW = width_meter(image.size(2))
        print('H: %.2f, W: %.2f' % (avgH, avgW))
