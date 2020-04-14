from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import time

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

#For testing 5 worst ious
  ious_dict = {}

#######################
  ious = []
  start_time = time.time()
  counter = 0
  for image_name in image_names:
      if counter<500:
          ret = detector.run(image_name)
          # print(time_str)
          # print(ret['iou'])
          ious.append(ret['iou'])
          ious_dict[image_name] = ret['iou']
          counter+=1
      else:
          break

  elapsed_time = time.time() - start_time
  print("Frame per second: ", 500/elapsed_time)
    # print(np.mean(np.array(ious)))
  print(sorted(ious_dict.items(), key=lambda kv: (kv[1], kv[0]))[0:10])
  l_np = np.array(ious)
  print("The mean is", np.mean(l_np))
  print("The standard deviation is", np.std(l_np))


if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
