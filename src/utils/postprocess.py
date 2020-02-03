import numpy as np

def Eason_eval_precision_recall(pred_BB, pred_score, true_BB, det_thresh):
    
    pred_hits = np.zeros(len(pred_BB))
    gt_hits = np.zeros(len(true_BB))
    hits_index = -np.ones(len(true_BB))
    hits_iou = np.zeros(len(true_BB), dtype=float)
    hits_score = np.zeros(len(true_BB), dtype=float)

    for pred_idx, pred_bb in enumerate(pred_BB):

        for gt_idx, gt_roi in enumerate(true_BB):
            pred_iou = Eason_iou(pred_bb[:6], gt_roi[:6])
            if pred_iou > det_thresh:
                gt_hits[gt_idx] = 1
                hits_index[gt_idx] = pred_idx
                hits_iou[gt_idx] = pred_iou
                hits_score[gt_idx]=pred_score[pred_idx]
                pred_hits[pred_idx] = 1

    TP = gt_hits.sum()
    FP = len(pred_hits) - pred_hits.sum()
    FN = len(true_BB)-gt_hits.sum()
    return int(TP),int(FP),int(FN),hits_index,hits_iou,hits_score