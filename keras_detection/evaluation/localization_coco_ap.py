"""
# adapted from https://github.com/rbgirshick/py-faster-rcnn
"""
from collections import defaultdict
from typing import List, Optional

import numpy as np

from keras_detection.evaluation.detection_metrics import Metric
from keras_detection.structures import LabelsFrame


def image_compute_coco_ap(
    target: LabelsFrame[np.ndarray],
    predicted: LabelsFrame[np.ndarray],
    ovthresholds: Optional[List[float]] = None,
) -> List[Metric]:

    num_targets = target.boxes.shape[0]
    gt_filenames = ["image"] * target.boxes.shape[0]
    det_filenames = ["image"] * predicted.boxes.shape[0]

    if ovthresholds is None:
        ovthresholds = np.arange(0.5, 1, 0.05)

    avg_ap, aps = eval_multi_threshold(
        gt_filenames,
        target.boxes_x1y1x2y2,
        det_filenames,
        predicted.boxes_x1y1x2y2,
        predicted.weights,
        ovthresholds=ovthresholds,
    )
    return [Metric("avg_AP", float(avg_ap), num_targets)] + [
        Metric(f"AP@{int(th*100)}", float(ap), num_targets)
        for th, ap in zip(ovthresholds, aps)
    ]


def eval_multi_threshold(
    gtFilenames,
    gtBBoxes,
    detFilenames,
    detBBoxes,
    detConfidences,
    ovthresholds=None,
    verbose: bool = False,
):
    """

    gtFilenames=gtFilenames[gtIndices],
    gtBBoxes=gtBBoxes[gtIndices],
    detFilenames=detFilenames[detIndices],
    detBBoxes=detBBoxes[detIndices],
    detConfidences=detConfidences[detIndices],

    Evaluate detection results as the average results of multiple overlap thresholds
    :param gtFilenames: list of n image ground-truth file names
    :param gtBBoxes: array of nX4 ground-truth bounding boxes
    :param detFilenames: list of m image detections' file names
    :param detBBoxes: array of mX4 detection bounding boxes
    :param detConfidences: m array of detection confidence scores
    :param ovthresholds: list of overlap thresholds
    :return:
    """
    if ovthresholds is None:
        ovthresholds = np.arange(0.5, 1, 0.05)

    if verbose:
        print("Evaluation results at thresholds {}".format(ovthresholds))
    aps = eval_thresholds_fast(
        gtFilenames,
        gtBBoxes,
        detFilenames,
        detBBoxes,
        detConfidences,
        ovthresholds=ovthresholds,
    )
    if verbose:
        print(f"APs {aps}")
    return np.mean(aps), aps


def voc_ap(rec, prec):
    """
    Compute AP
    :param rec: list of recall values at each detection
    :param prec: list of precision values at each detection
    :return:
    """
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_thresholds_fast(
    gt_imgs, gt_bboxes, det_imgs, det_bboxes, det_confidences, ovthresholds
):
    """
    # adapted from https://github.com/rbgirshick/py-faster-rcnn
    Evaluate detection results using a single overlap threshold
    :param gt_imgs: list of n image ground-truth file names
    :param gt_bboxes: array of nX4 ground-truth bounding boxes
    :param det_imgs: list of m image detections' file names
    :param det_bboxes: array of mX4 detection bounding boxes
    :param det_confidences: m array of detection confidence scores
    :param ovthresh: overlap threshold to consider a detection as correct
    :return:
    """
    if not set(det_imgs).issubset(set(gt_imgs)):
        raise Exception(
            "Error: detection results include images outside of the groundtruth annotations"
        )

    num_positives = len(gt_imgs)

    # keep gt data separately for each image. The data includes the object bounding bboxes (bboxes) and an indication
    # of which objects were detected (detected)
    gt_img_indices = defaultdict(list)
    for i, gt_img in enumerate(gt_imgs):
        gt_img_indices[gt_img].append(i)
    gt_img_data = {
        img: {
            "bboxes": gt_bboxes[indices],
            "detected": [[False] * len(indices) for k in range(len(ovthresholds))],
        }
        for img, indices in gt_img_indices.items()
    }

    # sort detections by decreasing confidence
    sorted_ind = np.argsort(-det_confidences)

    det_bboxes = det_bboxes[sorted_ind, :]
    det_imgs = [det_imgs[x] for x in sorted_ind]

    # iterate over detections and determine TPs and FPs
    tp = {}
    fp = {}
    for k, _ in enumerate(ovthresholds):
        tp[k] = np.zeros(len(det_bboxes))
        fp[k] = np.zeros(len(det_bboxes))

    jmax = -1
    for d, img_name in enumerate(det_imgs):
        bb = det_bboxes[d, :].astype(float)
        ovmax = -np.inf
        BBGT = gt_img_data[img_name]["bboxes"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        for k, ovthresh in enumerate(ovthresholds):
            if ovmax > ovthresh:
                if not gt_img_data[img_name]["detected"][k][jmax]:
                    tp[k][d] = 1.0
                    gt_img_data[img_name]["detected"][k][jmax] = True
                else:
                    fp[k][d] = 1.0
            else:
                fp[k][d] = 1.0

    # print("Computing VOC AP ...")
    aps = []
    for k, _ in enumerate(ovthresholds):
        # compute precision recall
        fp_th = np.cumsum(fp[k])
        tp_th = np.cumsum(tp[k])
        rec = tp_th / float(num_positives)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp_th / np.maximum(tp_th + fp_th, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec)
        aps.append(ap)

    return aps
