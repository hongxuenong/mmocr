from mmocr.utils.ocr import MMOCR
import cv2 as cv
import numpy as np
import os
from eval_det_iou import DetectionIoUEvaluator
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    # Load models into memory
    ocr = MMOCR(det='FCE_CTW_DCNv2', recog=None, device='cuda:1')

    evaluator = DetectionIoUEvaluator(iou_constraint=0.5)

    gtDir = 'data/eval_dataset_v1/det/annotations/test/'
    imgDir = 'data/eval_dataset_v1/det/imgs/test/'

    gtFiles = [name for name in os.listdir(gtDir) if name.endswith('.txt')]

    print(len(gtFiles))

    preds = []
    gts = []
    for name in tqdm(gtFiles):
        imagePath = imgDir + name.replace('txt', 'jpg')
        img = cv.imread(imagePath)
        try:
            boxes = ocr.readtext(imagePath)[0]['boundary_result']
        except:
            continue
        pred = []
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

        preds.append(pred)

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

        gts.append(gt)

    results = []
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    metrics = evaluator.combine_results(results)
    print(metrics)
