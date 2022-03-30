# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import random as rng
from shapely.geometry import Polygon
import pyclipper


def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


text_repr_type = 'poly'
mask_thr = 0.2 * 255
min_text_score = 0.3
min_text_width = 5
unclip_ratio = 1.5
max_candidates = 3000

prob_map = cv2.imread('outputs/DBNet_CTW/test/1005_binary.png', 0)
img = cv2.imread('outputs/DBNet_CTW/test/1005.jpg')
img = cv2.imread('outputs/DBNet_CTW/test/1005_binary.png')
# print(img)
text_mask = prob_map > mask_thr

# score_map = prob_map.data.cpu().numpy().astype(np.float32)
# text_mask = text_mask.data.cpu().numpy().astype(np.uint8)  # to numpy

contours, hierarchy = cv2.findContours((text_mask * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

boundaries = []

score_map = prob_map.astype(np.float32)
p = []
for i, poly in enumerate(contours):

    if i > max_candidates:
        break
    epsilon = 0.001 * cv2.arcLength(poly, True)
    approx = cv2.approxPolyDP(poly, epsilon, True)
    points = approx.reshape((-1, 2))
    if points.shape[0] < 4:
        print(approx.shape)
        print(points.shape)
        print(poly.shape)
        continue
    score = box_score_fast(score_map, points)
    if score < min_text_score:
        continue
    poly = unclip(points, unclip_ratio=unclip_ratio)
    # poly = unclip(poly, unclip_ratio=unclip_ratio)
    if len(poly) == 0 or isinstance(poly[0], list):
        continue
    # poly = poly.reshape(-1, 2)
    poly = poly.transpose(1, 0, 2)
    # print(poly.shape)
    # if self.text_repr_type == 'quad':
    #     poly = points2boundary(poly, self.text_repr_type, score,
    #                            self.min_text_width)
    # elif self.text_repr_type == 'poly':
    #     poly = poly.flatten().tolist()
    #     if score is not None:
    #         poly = poly + [score]
    #     if len(poly) < 8:
    #         poly = None
    if poly is not None:
        boundaries.append(poly)
print(len(boundaries))
print(len(contours))
for i in range(len(boundaries)):
    # print(poly.shape)
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv2.drawContours(img, boundaries, i, color, 2, cv2.LINE_8, hierarchy, 0)
cv2.imwrite('test.png', img)
cv2.imwrite('test_mask.png', text_mask * 255)