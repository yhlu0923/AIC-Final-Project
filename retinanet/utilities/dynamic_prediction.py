#! /usr/bin/env python3
import argparse
import cv2
import numpy as np
import skimage
import torch
from torchvision import transforms

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image, min_side=608, max_side=1024):
        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        return torch.from_numpy(new_image)

class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, image):
        return (image.astype(np.float32) - self.mean) / self.std

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def collater(imgs):
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return padded_imgs

def main(args=None):
  parser = argparse.ArgumentParser('Dynamic Prediction')
  parser.add_argument('--model', help='Path to the model (.pt) file')
  parser.add_argument('--video', help='Path to the video', default=0)

  parser = parser.parse_args(args)

  cap = cv2.VideoCapture(parser.video)
  if not cap.isOpened():
    print("error: video capture")
    return

  retinanet = torch.load(parser.model)
  if torch.cuda.is_available():
    retinanet = retinanet.cuda()

  retinanet.eval()

  transform = transforms.Compose([Normalizer(), Resizer()])
  unnormalize = UnNormalizer()

  if parser.video != 0:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,  480))

  frame_count = 0
  while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break

    if frame_count == 0:
      frame = frame.astype(np.float32) / 255.0
      frame = transform(frame)
      frame = collater([frame])

      if torch.cuda.is_available():
        scores, classification, transformed_anchors = retinanet(frame.cuda().float())
      else:
        scores, classification, transformed_anchors = retinanet(frame.float())

      idxs = np.where(scores.cpu() > 0.5)
      frame = np.array(255 * unnormalize(frame[0, :, :, :])).copy()

      frame[frame < 0] = 0
      frame[frame > 255] = 255

      frame = np.transpose(frame, (1, 2, 0))
      frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)

      for j in range(idxs[0].shape[0]):
        bbox = transformed_anchors[idxs[0][j], :]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        label = int(classification[idxs[0][j]])

        # without_mask
        if label == 0:
          cv2.rectangle(frame, (x1, y1), (x2, y2), color=(5, 13, 240), thickness=2)
        # with_mask
        elif label == 1:
          cv2.rectangle(frame, (x1, y1), (x2, y2), color=(31, 224, 89), thickness=2)
        # mask_wear_incorrect
        elif label == 2:
          cv2.rectangle(frame, (x1, y1), (x2, y2), color=(33, 174, 255), thickness=2)

    frame_count += 1
    if frame_count == 30:
      frame_count = 0

    if parser.video == 0:
      cv2.imshow('frame', frame)
    else:
      out.write(frame)

    if cv2.waitKey(1) == ord('q'):
      break

  # 釋放該攝影機裝置
  cap.release()
  if parser.video != 0:
    out.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
