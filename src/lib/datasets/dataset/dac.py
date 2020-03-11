from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data


class DAC(data.Dataset):
    num_classes = 96
    default_resolution = [640, 384]
    mean = np.array([0.44387722, 0.45443517, 0.41266478],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.2316625934896425, 0.21541203506039922, 0.24325553952463594],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(DAC, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'data')
        self.img_dir = os.path.join(self.data_dir, 'pictures', 'pictures')
        if split == 'test':
            self.annot_path = os.path.join(
                self.data_dir,
                'anno_test.json')
            print("test mode")
        else:
            # if opt.task == 'exdet':
            #     self.annot_path = os.path.join(
            #         self.data_dir,
            #         'anno.json')
            # else:
            self.annot_path = os.path.join(
                self.data_dir,
                'anno_train.json')
            print("Train mode")
        self.max_objs = 128
        self.class_name = [
            '__background__', 'boat1', 'boat2', 'boat3', 'boat4', 'boat5',
            'boat6', 'boat7', 'boat8', 'building1', 'building2',
            'building3', 'car1', 'car2', 'car3', 'car4',
            'car5', 'car6', 'car8', 'car9',
            'car10', 'car11', 'car12', 'car13', 'car14',
            'car15', 'car16', 'car17', 'car18', 'car19',
            'car20', 'car21', 'car22', 'car23', 'car24',
            'drone1', 'drone2', 'drone3', 'drone4', 'group2',
            'group3', 'horseride1', 'paraglider1', 'person1', 'person2',
            'person3', 'person4', 'person5', 'person6', 'person7',
            'person8', 'person9', 'person10', 'person11', 'person12',
            'person13', 'person14', 'person15', 'person16', 'person17',
            'person18', 'person19', 'person20', 'person21', 'person22',
            'person23', 'person24', 'person25', 'person26', 'person27',
            'person28', 'person29', 'riding1', 'riding2', 'riding3',
            'riding4', 'riding5', 'riding6', 'riding7', 'riding8',
            'riding9', 'riding10', 'riding11', 'riding12', 'riding13',
            'riding14', 'riding15', 'riding16', 'riding17', 'truck1',
            'truck2', 'wakeboard1', 'wakeboard2', 'wakeboard3', 'wakeboard4',
            'whale1']
        self._valid_ids =  np.arange(1, 96, dtype=np.int32)
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        # print(self.cat_ids)
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

