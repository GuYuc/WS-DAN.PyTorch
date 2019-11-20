"""EVALUATION
Created: Apr 22,2019 - Yuchong Gu
Revised: Nov 20,2019 - Yuchong Gu
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pdb
import sys
import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from models import WSDAN
from utils import TopKAccuracyMetric, batch_augment
from datasets import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# visualize
visualize = False

ToPILImage = transforms.ToPILImage()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

savepath = '/home/guyuchong/DATA/FGVC/StanfordCars/visualize'
os.makedirs(savepath, exist_ok=True)


def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)


def main():
    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    try:
        ckpt = sys.argv[1]
    except:
        logging.info('Usage: python3 eval.py <model.ckpt>')
        return

    ##################################
    # Dataset for testing
    ##################################
    test_dataset = CarDataset(phase='test', resize=448)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    ##################################
    # Initialize model
    ##################################
    net = WSDAN(num_classes=test_dataset.num_classes, M=32, net='inception_mixed_6e')

    # Load ckpt and get state_dict
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    net.load_state_dict(state_dict)
    logging.info('Network loaded from {}'.format(ckpt))

    ##################################
    # use cuda
    ##################################
    cudnn.benchmark = True
    net.to(device)
    net = nn.DataParallel(net)
    net.eval()

    ##################################
    # Prediction
    ##################################
    accuracy = TopKAccuracyMetric(topk=(1, 5))
    accuracy.reset()

    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), unit=' batches')
        pbar.set_description('Validation')
        for i, (X, y) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)

            # WS-DAN
            y_pred_raw, feature_matrix, attention_maps = net(X)

            # Augmentation with crop_mask
            crop_image = batch_augment(X, attention_maps, mode='crop', theta=0.1)

            y_pred_crop, _, _ = net(crop_image)
            pred = (y_pred_raw + y_pred_crop) / 2.

            if visualize:
                # reshape attention maps
                attention_maps = F.upsample_bilinear(attention_maps, size=(X.size(2), X.size(3)))
                attention_maps = torch.sqrt(attention_maps.cpu() / attention_maps.max().item())

                # get heat attention maps
                heat_attention_maps = generate_heatmap(attention_maps)

                # raw_image, heat_attention, raw_attention
                raw_image = X.cpu() * STD + MEAN
                heat_attention_image = raw_image * 0.5 + heat_attention_maps * 0.5
                raw_attention_image = raw_image * attention_maps

                for batch_idx in range(X.size(0)):
                    rimg = ToPILImage(raw_image[batch_idx])
                    raimg = ToPILImage(raw_attention_image[batch_idx])
                    haimg = ToPILImage(heat_attention_image[batch_idx])
                    rimg.save(os.path.join(savepath, '%03d_raw.jpg' % (i + batch_idx)))
                    raimg.save(os.path.join(savepath, '%03d_raw_atten.jpg' % (i + batch_idx)))
                    haimg.save(os.path.join(savepath, '%03d_heat_atten.jpg' % (i + batch_idx)))

            # Top K
            epoch_acc = accuracy(pred, y)

            # end of this batch
            batch_info = 'Val Acc ({:.2f}, {:.2f})'.format(epoch_acc[0], epoch_acc[1])
            pbar.update()
            pbar.set_postfix_str(batch_info)

        pbar.close()

    # show information for this epoch
    logging.info('Accuracy: %.2f, %.2f' % (epoch_acc[0], epoch_acc[1]))


if __name__ == '__main__':
    main()
