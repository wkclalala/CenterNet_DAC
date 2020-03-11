from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory
import numpy as np

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  save_dir = opt.demo
  image_names = []
  with open(save_dir, 'r') as sd:
      lines = sd.readlines()
      lines = lines[0].split(".jpg ")
      for line in lines[:-1]:
          image_names.append(os.path.join('../data/data/pictures/pictures/' + line + ".jpg"))

  ious = []
  for image_name in image_names:
    print(image_name)
    ret = detector.run(image_name)
    time_str = ''
    for stat in time_stats:
      time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
    # print(time_str)
    # print(ret['iou'])
    ious.append(ret['iou'])
    # print(np.mean(np.array(ious)))

  l_np = np.array(ious)
  print("The mean is", np.mean(l_np))
  print("The standard deviation is", np.std(l_np))
  with open('../data/data/listfile.txt', 'w') as filehandle:
      for iou in ious:
          filehandle.write(np.array2string(iou)+"\n")


if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
