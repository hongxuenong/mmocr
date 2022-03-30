import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from mmocr.utils.ocr import MMOCR
import cv2 as cv
import numpy as np
import os
from eval_det_iou import DetectionIoUEvaluator
import matplotlib.pyplot as plt
from tqdm import tqdm
import random as rng
import math
# from mmocr.utils import stitch_boxes_into_lines

eps = 1e-10


def is_on_same_line(box_a, box_b, min_y_overlap_ratio=0.8):
    """Check if two boxes are on the same line by their y-axis coordinates.

    Two boxes are on the same line if they overlap vertically, and the length
    of the overlapping line segment is greater than min_y_overlap_ratio * the
    height of either of the boxes.

    Args:
        box_a (list), box_b (list): Two bounding boxes to be checked
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                                    allowed for boxes in the same line

    Returns:
        The bool flag indicating if they are on the same line
    """
    a_y_min = np.min(box_a[1::2])
    b_y_min = np.min(box_b[1::2])
    a_y_max = np.max(box_a[1::2])
    b_y_max = np.max(box_b[1::2])

    # Make sure that box a is always the box above another
    if a_y_min > b_y_min:
        a_y_min, b_y_min = b_y_min, a_y_min
        a_y_max, b_y_max = b_y_max, a_y_max

    if b_y_min <= a_y_max:
        if min_y_overlap_ratio is not None:
            sorted_y = sorted([b_y_min, b_y_max, a_y_max])
            overlap = sorted_y[1] - sorted_y[0]
            min_a_overlap = (a_y_max - a_y_min) * min_y_overlap_ratio
            min_b_overlap = (b_y_max - b_y_min) * min_y_overlap_ratio
            return overlap >= min_a_overlap or \
                overlap >= min_b_overlap
        else:
            return True
    return False


def get_line(bbox):

    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
    length = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    width = np.sqrt((x1 - x4)**2 + (y1 - y4)**2)
    if length > width:
        x_mean_1 = (x1 + x4) / 2
        y_mean_1 = (y1 + y4) / 2
        x_mean_2 = (x2 + x3) / 2
        y_mean_2 = (y2 + y3) / 2
        k = (y_mean_1 - y_mean_2) / [(x_mean_1 - x_mean_2) + eps]
        b = y_mean_1 - k * x_mean_1
    else:
        x_mean_1 = (x1 + x2) / 2
        y_mean_1 = (y1 + y2) / 2
        x_mean_2 = (x3 + x4) / 2
        y_mean_2 = (y3 + y4) / 2
        k = (y_mean_1 - y_mean_2) / [(x_mean_1 - x_mean_2) + eps]

        b = y_mean_1 - k * x_mean_1
    center = ((x1 + x2 + x4 + x4) / 4, (y1 + y2 + y3 + y4) / 4)
    return k, b, width, length, center


def is_on_same_line_new(box_a, box_b, grad_thresh=5):
    """Check if two boxes are on the same line.

    Two boxes are on the same line if:
    1) they have small difference in gradient k
    2) they have small distance (assuming same gradient)
    
    Args:
        box_a (list), box_b (list): Two bounding boxes to be checked
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                                    allowed for boxes in the same line

    Returns:
        The bool flag indicating if they are on the same line
    """
    k_a, b_a, width_a, length_a, center_a = get_line(box_a)
    k_b, b_b, width_b, length_b, center_b = get_line(box_b)
    short_edge_a = min(width_a, length_a)
    short_edge_b = min(width_b, length_b)
    short_edge = min(short_edge_a, short_edge_b)
    dist_threshold = short_edge / 2

    d_a = math.degrees(np.arctan(k_a))
    d_b = math.degrees(np.arctan(k_b))

    if np.abs(d_a - d_b) < grad_thresh:
        distance_line = np.abs(b_a - b_b) / np.sqrt(
            1 +
            k_a**2)  # distance between two parallel line: for inclined cases
        distance_y = np.abs(center_a[1] -
                            center_b[1])  # distance on vertical direction
        distance = min(distance_line, distance_y)
        # print("distance:", distance)
        # print("dist_threshold:", dist_threshold)
        if distance < dist_threshold:
            # print("box_a:", box_a)
            # print("box_b:", box_b)
            # print(k_a, k_b, d_a, d_b, b_a, b_b)
            return True
    else:
        return False


def visualize(img, bboxes, savepath, color=[255, 0, 0], thickness=2):
    for bbox in bboxes:
        bbox = bbox['points']
        bbox = np.array(bbox).astype(np.int32)
        cv.polylines(
            img, [bbox[:8].reshape(-1, 2)],
            True,
            color=color,
            thickness=thickness)
    cv.imwrite(savepath, img)
    return img


def filter_small_boxes(bboxes, thre=12):
    bbox_to_keep = []
    for bbox in bboxes:
        bbox = bbox['points']
        length = np.sqrt((bbox[0][0] - bbox[1][0])**2 +
                         (bbox[0][1] - bbox[1][1])**2)
        width = np.sqrt((bbox[0][0] - bbox[3][0])**2 +
                        (bbox[0][1] - bbox[3][1])**2)
        if (length > thre) and (width > thre):
            bbox_to_keep.append({
                'points': bbox,
                'ignore': False,
            })

    return bbox_to_keep


def stitch_boxes_into_lines(boxes, max_x_dist=20, min_y_overlap_ratio=0.8):
    """Stitch fragmented boxes of words into lines.

    Note: part of its logic is inspired by @Johndirr
    (https://github.com/faustomorales/keras-ocr/issues/22)

    Args:
        boxes (list): List of ocr results to be stitched
        max_x_dist (int): The maximum horizontal distance between the closest
                    edges of neighboring boxes in the same line
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                    allowed for any pairs of neighboring boxes in the same line

    Returns:
        merged_boxes(list[dict]): List of merged boxes and texts
    """

    if len(boxes) <= 1:
        return boxes

    merged_boxes = []

    boxes = [np.array(b['points']).reshape(-1) for b in boxes]
    # sort groups based on the x_min coordinate of boxes
    # print(boxes)
    x_sorted_boxes = sorted(boxes, key=lambda x: np.min(x[::2]))

    # print(x_sorted_boxes)
    # store indexes of boxes which are already parts of other lines

    skip_idxs = set()

    i = 0
    # locate lines of boxes starting from the leftmost one
    for i in range(len(x_sorted_boxes)):
        if i in skip_idxs:
            continue
        # the rightmost box in the current line
        rightmost_box_idx = i
        line = [rightmost_box_idx]
        for j in range(i + 1, len(x_sorted_boxes)):
            if j in skip_idxs:
                continue
            if is_on_same_line_new(x_sorted_boxes[rightmost_box_idx],
                                   x_sorted_boxes[j]):
                line.append(j)
                skip_idxs.add(j)
                rightmost_box_idx = j

        # split line into lines if the distance between two neighboring
        # sub-lines' is greater than max_x_dist
        lines = []
        line_idx = 0
        lines.append([line[0]])
        for k in range(1, len(line)):
            curr_box = x_sorted_boxes[line[k]]
            prev_box = x_sorted_boxes[line[k - 1]]
            dist = np.min(curr_box[::2]) - np.max(prev_box[::2])
            if dist > max_x_dist:
                line_idx += 1
                lines.append([])
            lines[line_idx].append(line[k])

        # Get merged boxes
        for box_group in lines:
            merged_box = {}
            # merged_box['text'] = ' '.join(
            #     [x_sorted_boxes[idx]['text'] for idx in box_group])
            if len(box_group) > 1:
                points = []
                for i, idx in enumerate(box_group):
                    for j in range(4):
                        points.append((x_sorted_boxes[idx][2 * j],
                                       x_sorted_boxes[idx][2 * j + 1]))

                rect = cv.minAreaRect(np.int0(points))
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = cv.boxPoints(rect)
            else:
                x1, y1, x2, y2, x3, y3, x4, y4 = x_sorted_boxes[box_group[0]]
            merged_box['points'] = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            merged_box['ignore'] = False
            merged_boxes.append(merged_box)
    return merged_boxes


def post_processing(pred):
    pred = filter_small_boxes(pred)
    pred = stitch_boxes_into_lines(pred)
    return pred
