""" NOTICE: A Custom Dataset SHOULD BE PROVIDED
Created: May 02,2019 - Yuchong Gu
Revised: May 07,2019 - Yuchong Gu
"""
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

__all__ = ['CustomDataset']


config = {
    # e.g. train/val/test set should be located in os.path.join(config['datapath'], 'train/val/test')
    'datapath': 'DATA_PATH',
}


class CustomDataset(Dataset):
    """
    # Description:
        Basic class for retrieving images and labels

    # Member Functions:
        __init__(self, phase, shape):   initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            shape:                      output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', shape=(512, 512)):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.data_path = os.path.join(config['datapath'], phase)
        self.data_list = os.listdir(self.data_path)

        self.shape = shape
        self.config = config

        # transform
        self.transform = transforms.Compose([
            transforms.Resize(size=(self.shape[0], self.shape[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.data_path, self.data_list[item])).convert('RGB')  # (C, H, W)
        image = self.transform(image)
        assert image.size(1) == self.shape[0] and image.size(2) == self.shape[1]

        if self.phase != 'test':
            # filename of image should have 'id_label.jpg/png' form
            label = int((self.data_list[item].split('.')[0]).split('_')[-1])  # label
            return image, label
        else:
            # filename of image should have 'id.jpg/png' form, and simply return filename in case of 'test'
            return image, self.data_list[item]

    def __len__(self):
        return len(self.data_list)
