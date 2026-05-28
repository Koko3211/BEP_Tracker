import numpy as np

"""
The following code was adapted from the SU-T tracker association file (https://github.com/vranlee/SU-T) for
the ablation study.
"""
def hmiou(bboxes1, bboxes2):
    """
    Height_Modulated_IoU
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    yy11 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    yy12 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    yy21 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    yy22 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    o = (yy12 - yy11) / (yy22 - yy21)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o *= wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return (o)

def giou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)  

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1 
    hc = yyc2 - yyc1 
    assert((wc > 0).all() and (hc > 0).all())
    area_enclose = wc * hc 
    giou = iou - (area_enclose - wh) / area_enclose
    giou = (giou + 1.)/2.0 # resize from (-1,1) to (0,1)
    return giou


def fish_iou_batch(bboxes1, bboxes2):
    """
    FishIoU
    """
    # Expand dimensions for batch calculation
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # Calculate standard intersection parameters
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    # Standard IoU calculation with epsilon for numerical stability
    eps = 1e-6
    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - wh

    iou = wh / (union + eps)

    # Calculate centers
    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    # Distance between centers
    center_dist = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    # Calculate diagonal length of enclosing box (for normalization)
    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    diagonal_length = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2 + eps

    # Center distance penalty (normalized by diagonal)
    center_penalty = center_dist / diagonal_length

    # Fish body shape is typically elongated - adjust central region
    # Fish head/body is often the front 70% of the bounding box
    # For simplicity, we'll define a smaller central region (30% inset)
    # that's more elongated horizontally (fish direction)
    cxx1 = bboxes1[..., 0] + 0.15 * w1  # 15% inset from left (potentially head)
    cyy1 = bboxes1[..., 1] + 0.3 * h1  # 30% inset from top
    cxx2 = bboxes1[..., 2] - 0.25 * w1  # 25% inset from right (potentially tail)
    cyy2 = bboxes1[..., 3] - 0.3 * h1  # 30% inset from bottom

    cxx1_2 = bboxes2[..., 0] + 0.15 * w2
    cyy1_2 = bboxes2[..., 1] + 0.3 * h2
    cxx2_2 = bboxes2[..., 2] - 0.25 * w2
    cyy2_2 = bboxes2[..., 3] - 0.3 * h2

    # Calculate central region intersection
    cxx1 = np.maximum(cxx1, cxx1_2)
    cyy1 = np.maximum(cyy1, cyy1_2)
    cxx2 = np.minimum(cxx2, cxx2_2)
    cyy2 = np.minimum(cyy2, cyy2_2)
    cw = np.maximum(0., cxx2 - cxx1)
    ch = np.maximum(0., cyy2 - cyy1)
    central_overlap = cw * ch

    # Calculate central region of each box for normalization
    central_area1 = (cxx2 - cxx1) * (cyy2 - cyy1) + eps
    central_area2 = (cxx2_2 - cxx1_2) * (cyy2_2 - cyy1_2) + eps
    central_union = central_area1 + central_area2 - central_overlap + eps

    # Central IoU
    ciou = central_overlap / central_union

    # Calculate aspect ratio consistency
    # For fish, width/height ratio is important for orientation
    ar1 = w1 / (h1 + eps)
    ar2 = w2 / (h2 + eps)
    ar_min = np.minimum(ar1, ar2)
    ar_max = np.maximum(ar1, ar2)
    ar_consistency = ar_min / (ar_max + eps)

    # Area ratio consistency - fish don't rapidly change size
    area_ratio = np.minimum(area1, area2) / (np.maximum(area1, area2) + eps)

    # Scale penalty - special consideration for small objects
    scale_factor = 1.0 - np.exp(-(np.minimum(area1, area2) / 1000.0))  # Reduce penalty for very small objects

    # Combine all components with specific weights for fish
    iou_weight = 1.0
    ciou_weight = 0.3
    ar_weight = 0.1
    area_ratio_weight = 0.2
    center_weight = 0.4 * scale_factor  # Less center emphasis for very small fish

    fish_iou = (iou_weight * iou +
                ciou_weight * ciou +
                ar_weight * ar_consistency +
                area_ratio_weight * area_ratio -
                center_weight * center_penalty)

    # Normalize to [0,1] range
    fish_iou = np.clip(fish_iou, 0, 1)

    return fish_iou

def ciou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh) 

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    
    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    # prevent dividing over zero. add one pixel shift
    h2 = h2 + 1.
    h1 = h1 + 1.
    arctan = np.arctan(w2/h2) - np.arctan(w1/h1)
    v = (4 / (np.pi ** 2)) * (arctan ** 2)
    S = 1 - iou 
    alpha = v / (S+v)
    ciou = iou - inner_diag / outer_diag - alpha * v
    
    return (ciou + 1) / 2.0 # resize from (-1,1) to (0,1)