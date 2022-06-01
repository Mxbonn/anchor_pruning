import numpy as np

def iou(bbox_a, bbox_b):
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    intersection = w * h
    bbox_a_area = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    bbox_b_area = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

    interounion = intersection / (bbox_a_area + bbox_b_area - intersection)

    return interounion


def iou_matching(dets, gt, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])

    xx1 = np.maximum(gt[0], x1[:])
    yy1 = np.maximum(gt[1], y1[:])
    xx2 = np.minimum(gt[2], x2[:])
    yy2 = np.minimum(gt[3], y2[:])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    ovr = inter / (gt_area + areas[:] - inter)

    inds = np.where(ovr >= thresh)[0]

    return inds, ovr[inds]