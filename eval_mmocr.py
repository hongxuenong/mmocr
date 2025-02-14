from dis import dis
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


if __name__ == '__main__':
    # Load models into memory
    # ocr = MMOCR(det='FCE_CTW_DCNv2', recog=None, device='cuda:0')
    ocr = MMOCR(
        det_config='configs/textdet/custom/dbnet_swinbase_eval_dataset_v1.py',
        det_ckpt='work_dirs/dbnet_swinbase_eval_dataset_v1/latest.pth',
        recog=None,
        device='cuda:0')

    evaluator = DetectionIoUEvaluator(iou_constraint=0.5)

    # gtDir = '../data/syn+MTWI2018/annotations/test/'
    # imgDir = '../data/syn+MTWI2018/imgs/test/'
    gtDir = '../data/bm_evl_cv_1k_en/annotations/test/'
    imgDir = '../data/bm_evl_cv_1k_en/imgs/test/'

    # gtDir = '../data/adhoc/annotations/'
    # imgDir = '../data/adhoc/imgs/'

    gtFiles = [name for name in os.listdir(gtDir) if name.endswith('.txt')]

    print("number of files:", len(gtFiles))

    preds = []
    preds_processed = []
    gts = []
    good = []
    bad = []
    improved = []
    no_improved = []
    ext_bad = []
    for name in tqdm(gtFiles):
        imagePath = imgDir + name.replace('txt', 'jpg')
        img = cv.imread(imagePath)
        # outs = ocr.readtext(imagePath)[0]
        try:
            outs = ocr.readtext(imagePath)[0]
        except:
            continue

        pred = []
        boxes = outs['boundary_result']
        for k, box in enumerate(boxes):
            points = []
            for j in range(int(len(box) / 2)):
                pA = box[j * 2] if box[j * 2] >= 0 else 0
                pB = box[j * 2 + 1] if box[j * 2 + 1] >= 0 else 0
                points.append((pA, pB))
            rect = cv.minAreaRect(np.int0(points))
            points = cv.boxPoints(rect)
            pred.append({
                'points': points,
                'ignore': False,
            })
        pred_processed = post_processing(pred)

        preds.append(pred)
        preds_processed.append(pred_processed)
        gt = []
        gtlines = []
        with open(gtDir + name) as fr:
            gtlines = fr.readlines()
        for line in gtlines:
            dataSplit = line.strip().split(',')
            x1, y1, x2, y2, x3, y3, x4, y4 = [float(_) for _ in dataSplit[:8]]
            gtTxt = ','.join(dataSplit[8:])
            if gtTxt == '###':
                continue
            gt.append({
                'points': [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                'ignore': False,
            })
        # gt = filter_small_boxes(gt)
        gt = post_processing(gt)

        gts.append(gt)

        test = evaluator.evaluate_image(gt, pred)
        precision, recall, hmean = test['precision'], test['recall'], test[
            'hmean']
        test_processed = evaluator.evaluate_image(gt, pred_processed)
        precision, recall, hmean = test_processed['precision'], test_processed[
            'recall'], test_processed['hmean']
        img_savename = imagePath.split('/')[-1].split(
            '.'
        )[0] + '_p' + f'{precision:.2f}' + '_r' + f'{recall:.2f}' + '_h' + f'{hmean:.2f}' + '.png'

        if (hmean > 0.7):
            good.append(hmean)
            savepath = 'outputs/bm_evl_cv_1k_en/good/' + img_savename
        elif hmean < 0.1:
            ext_bad.append(hmean)
            savepath = 'outputs/bm_evl_cv_1k_en/extremely_bad/' + img_savename
        else:
            bad.append(hmean)
            savepath = 'outputs/bm_evl_cv_1k_en/bad/' + img_savename
        # savepath = 'outputs/adhoc/' + img_savename
        img = visualize(img, pred, savepath)
        img = visualize(img, gt, savepath, color=[0, 255, 0])
        img = visualize(img, pred_processed, savepath, color=[0, 0, 255])

        # if (test_processed['hmean'] < hmean):
        #     no_improved.append(hmean)
        #     drop = hmean - test_processed['hmean']
        #     savepath = 'outputs/test/drops/' + str(drop) + '_' + img_savename

        # else:
        #     improved.append(test_processed['hmean'])
        #     savepath = 'outputs/test/improved/' + img_savename
        # img = cv.imread(imagePath)
        # img = visualize(img, pred, savepath)
        # img = visualize(img, gt, savepath, color=[0, 255, 0])
        # img = visualize(
        #     img, pred_processed, savepath, color=[0, 0, 255], thickness=1)

    print("number of good:", len(good))
    print("number of bad:", len(bad))
    print("number of extremely bad:", len(ext_bad))
    print("number of improves:", len(improved))
    print("number of drops:", len(no_improved))

    print(np.average(np.array(good)))
    print(np.average(np.array(bad)))
    results = []
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print("raw pred:", metrics)

    results = []
    for gt, pred_processed in zip(gts, preds_processed):
        results.append(evaluator.evaluate_image(gt, pred_processed))
    metrics = evaluator.combine_results(results)
    print("processed pred:", metrics)
