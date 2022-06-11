import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
  UnNormalizer, Normalizer


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
  parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

  parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
  parser.add_argument('--coco_path', help='Path to COCO directory')
  parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
  parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

  parser.add_argument('--model', help='Path to model (.pt) file.')

  parser = parser.parse_args(args)

  if parser.dataset == 'coco':
    dataset_val = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Resizer()]))
  elif parser.dataset == 'csv':
    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
  else:
    raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

  sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
  dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

  retinanet = torch.load(parser.model)

  use_gpu = True

  if use_gpu:
    if torch.cuda.is_available():
      retinanet = retinanet.cuda()

  if torch.cuda.is_available():
    retinanet = torch.nn.DataParallel(retinanet).cuda()
  else:
    retinanet = torch.nn.DataParallel(retinanet)

  retinanet.eval()

  total_time = 0
  img_num = 0

  for data in dataloader_val:

    with torch.no_grad():
      st = time.time()
      if torch.cuda.is_available():
        _, _, _ = retinanet(data['img'].cuda().float())
      else:
        _, _, _ = retinanet(data['img'].float())
      elapsed_time = time.time()-st

      print('Elapsed time: {}'.format(elapsed_time))

      total_time += elapsed_time
      img_num += 1

  print(f'average speed: { total_time / img_num }')

if __name__ == '__main__':
 main()
