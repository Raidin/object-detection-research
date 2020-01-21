import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

from matplotlib import patches


def SelectiveSearch(img):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rois = ss.process()

    # selective search의 result format = x, y, w, h
    # system에서 사용되는 format = x1, y1, x2, y2
    rois[:, 2] = rois[:, 0] + rois[:, 2]
    rois[:, 3] = rois[:, 1] + rois[:, 3]

    return rois

def ComputeIOU(gt, p):

    x1 = np.maximum(gt[0], p[:, 0])
    y1 = np.maximum(gt[1], p[:, 1])
    x2 = np.minimum(gt[2], p[:, 2])
    y2 = np.minimum(gt[3], p[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
    propoesed_area = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    union = gt_area + propoesed_area[:] - intersection[:]

    iou = intersection/union

    return iou

def ComputeTargetDelta(gt, p):
    d = np.zeros_like(p, dtype=np.float32)
    gt = gt.astype(np.float32)
    p = p.astype(np.float32)

    d[:, 0] = np.divide(np.subtract(gt[0], p[:,0]), p[:,2])
    d[:, 1] = np.divide(np.subtract(gt[1], p[:,1]), p[:,3])
    d[:, 2] = np.log(np.divide(gt[2], p[:,2]))
    d[:, 3] = np.log(np.divide(gt[2], p[:,3]))

    return d

def SeparateToRegions(gt_bbox, rois, th=0.5):
    regions = np.array([], dtype=np.float32).reshape(0, 5)
    deltas = np.array([], dtype=np.float32).reshape(0, 4)

    pos_candidate = np.array([], dtype=np.float32).reshape(0, 5)
    rois_copy = rois.copy()

    for gt in gt_bbox:
        sub_regions = np.array([], dtype=np.float32).reshape(0, 5)

        iou = ComputeIOU(gt, rois_copy)

        # ADD Candidate Positive Regions(IOU > 0.5)
        idx = np.where(iou > th)[0]
        sub_regions = np.vstack([sub_regions, np.column_stack([rois_copy[idx], iou[idx]])])
        rois_copy = np.delete(rois_copy, idx, axis=0)

        # ADD GT
        gt = np.insert(gt, 4, 1.0)
        sub_regions = np.vstack([sub_regions, gt])
        sub_delta = ComputeTargetDelta(gt, sub_regions)

        deltas = np.vstack([deltas, np.delete(sub_delta, 4, axis=1)])
        pos_candidate = np.vstack([pos_candidate, sub_regions])

    # Positive Regions Sort(Descending)

    sort_dix = np.argsort(-pos_candidate[:, 4])
    pos_candidate = pos_candidate[sort_dix][:32]
    pos_candidate = np.delete(pos_candidate, 4, axis=1)
    deltas = deltas[:32]

    # ADD Positive Regions
    regions = np.vstack([regions, np.insert(pos_candidate[:], 4, 1, axis=1)])

    # ADD Negative Regions
    neg_num = pos_candidate.shape[0] * 3
    if rois_copy.shape[0] > neg_num:
        neg_idx = random.sample(range(rois_copy.shape[0]), neg_num)
        rois_copy = rois_copy[neg_idx]
    regions = np.vstack([regions, np.insert(rois_copy[:], 4, 0, axis=1)])

    return regions, deltas

def WarppingImage(img, regions, delta):
    cls_trn_img = np.array([], dtype=np.uint8).reshape(0, 224, 224, 3)
    cls_trn_lb = np.array([], dtype=np.int32).reshape(0, 1)
    reg_trn_img = np.array([], dtype=np.uint8).reshape(0, 224, 224, 3)
    reg_trn_delta = np.array([], dtype=np.float32).reshape(0, 4)
    cnt = 0

    for region in regions:
        x1, y1, x2, y2, label = region

        # 원본영상에서 region 영역 crop
        timg = img[int(y1):int(y2), int(x1):int(x2)]
        # 224x224 크기로 wrapping
        rimg = cv2.resize(timg, (224, 224), interpolation=cv2.INTER_AREA)
        # img file 그룹화 하기 위해서 dim 추가
        rimg = np.expand_dims(rimg, axis=0)

        cls_trn_img = np.vstack([cls_trn_img, rimg])
        cls_trn_lb = np.vstack([cls_trn_lb, label])

        if label == 1:
            reg_trn_img = np.vstack([reg_trn_img, rimg])
            reg_trn_delta = np.vstack([reg_trn_delta, delta[cnt]])
            cnt += 1

    return cls_trn_img, cls_trn_lb, reg_trn_img, reg_trn_delta

# Bounding Box Visualization
'''
@ img : numpy[x, y, c], opencv img read
@ title : image description, default :: 'Empty'
@ title : rectangle color, default :: 'magenta'
@ ax : plot axis object
'''
def DrawBoxes(img, bboxes, title='Empty', color='magenta', linestyle="solid", ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 10))

    # BBox Display
    # Box 좌표 구성(xmin, ymin, xmax, ymax)
    for bbox in bboxes:
        p = patches.Rectangle((bbox[0], bbox[1]), (bbox[2] - bbox[0]), (bbox[3] - bbox[1]), linewidth=2, alpha=1.0, linestyle=linestyle, edgecolor=color, facecolor='none')
        ax.add_patch(p)

    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    ax.set_title(title, color='white')

