from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import os
import json

try:
  from external.nms import soft_nms
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

from thop import profile

class CtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetDetector, self).__init__(opt)
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      # macs, params = profile(self.model, inputs=(images,))
      # print("macs: ",macs)
      # print("params: ", params)
      output = self.model(images)[-1]
      hm = output['hm'].sigmoid_()
      wh = output['wh']
      reg = output['reg'] if self.opt.reg_offset else None
      if self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg = reg[0:1] if reg is not None else None
      torch.cuda.synchronize()
      forward_time = time.time()
      dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
      
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))

  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    anno_path = os.path.join(self.opt.data_dir, 'data/box_test.json')
    file_name = self.image_path.split("/")[-1]
    print(file_name)
    ground_truth = []
    with open(anno_path) as json_file:
        data = json.load(json_file)
        ground_truth = data[file_name]

    iou = 0
    max_idx = 0
    max_confidence = 0
    max_cat = 0
    # print("threshold: ", self.opt.vis_thresh)
    for j in range(1, self.num_classes + 1):
      i = 0
      for bbox in results[j]:
        # print("Confidence: ", bbox[4])
        if bbox[4]>max_confidence:
          max_confidence = bbox[4]
          max_idx = i
          max_cat = j
        i+=1
    # print("Print image")
    # print(iou)
    # print(max_cat)
    # print(max_idx)
    # print(max_confidence)
    bbox_s = results[max_cat][max_idx]
    # print(bbox_s)
    # print(ground_truth)
    point_1 = [max(ground_truth[0], bbox_s[0]), max(ground_truth[1], bbox_s[1])]
    point_2 = [min(ground_truth[2] + ground_truth[0], bbox_s[2]),
               min(ground_truth[3] + ground_truth[1], bbox_s[3])]
    small_area = (point_2[1] - point_1[1]) * (point_2[0] - point_1[0])
    iou = small_area / (
              (ground_truth[2] * ground_truth[3]) + (bbox_s[3] - bbox_s[1]) * (bbox_s[2] - bbox_s[0]) - small_area)
    if iou < 0:
      iou = 0
    debugger.add_coco_bbox(
      [ground_truth[0], ground_truth[1], ground_truth[0] + ground_truth[2], ground_truth[1] + ground_truth[3]],
      0, 1, False, img_id='ctdet')
    debugger.add_coco_bbox(bbox_s[:4], 1, bbox_s[4], True, img_id='ctdet', iou="{:.3f}".format(iou))
    if(self.opt.save_image):
      debugger.save_all_imgs(path='/home/kw573/work/CenterNet/exp/ctdet/default',genID=True)
    return iou