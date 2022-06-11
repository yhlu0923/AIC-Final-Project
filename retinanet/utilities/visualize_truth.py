import numpy as np
import time
import time
import argparse

import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, \
  UnNormalizer, Normalizer

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
  parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

  parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
  parser.add_argument('--coco_path', help='Path to COCO directory')
  parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
  parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

  parser = parser.parse_args(args)

  if parser.dataset == 'coco':
    dataset_val = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Resizer()]))
  elif parser.dataset == 'csv':
    dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
  else:
    raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

  sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
  dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

  unnormalize = UnNormalizer()

  for data in dataloader_val:

    with torch.no_grad():
      img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

      img[img<0] = 0
      img[img>255] = 255

      img = np.transpose(img, (1, 2, 0))

      img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

      for annot in data['annot'][0]:
        x1 = int(annot[0].item())
        y1 = int(annot[1].item())
        x2 = int(annot[2].item())
        y2 = int(annot[3].item())
        label = int(annot[4].item())

        # without_mask
        if label == 0:
          cv2.rectangle(img, (x1, y1), (x2, y2), color=(5, 13, 240), thickness=2)
        # with_mask
        elif label == 1:
          cv2.rectangle(img, (x1, y1), (x2, y2), color=(31, 224, 89), thickness=2)
        # mask_wear_incorrect
        elif label == 2:
          cv2.rectangle(img, (x1, y1), (x2, y2), color=(33, 174, 255), thickness=2)

      cv2.imshow('img', img)
      key = cv2.waitKey(0)
      if key == ord('q') or key == 27: # esc
        break



if __name__ == '__main__':
 main()
