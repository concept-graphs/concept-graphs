import numpy as np
import torch


def compute_pred_gt_associations(pred, gt):
    # pred: predicted pointcloud
    # gt: GT pointcloud
    from chamferdist.chamfer import knn_points

    # pred = pointclouds.points_padded.cuda().contiguous()
    # gt = pts_gt.unsqueeze(0).cuda().contiguous()
    b, l, d = pred.shape
    lengths_src = torch.ones(b, dtype=torch.long, device=pred.device) * l
    b, l, d = gt.shape
    lengths_tgt = torch.ones(b, dtype=torch.long, device=pred.device) * l
    src_nn = knn_points(
        pred,
        gt,
        lengths1=lengths_src,
        lengths2=lengths_tgt,
        return_nn=True,
        return_sorted=True,
        K=1,
    )
    idx_pred_to_gt = src_nn.idx.squeeze(0).squeeze(-1)
    tgt_nn = knn_points(
        gt,
        pred,
        lengths1=lengths_tgt,
        lengths2=lengths_src,
        return_nn=True,
        return_sorted=True,
        K=1,
    )
    idx_gt_to_pred = tgt_nn.idx.squeeze(0).squeeze(-1)

    return idx_pred_to_gt, idx_gt_to_pred

def compute_confmatrix(
    labels_pred, labels_gt, idx_pred_to_gt, idx_gt_to_pred, class_names
):
    labels_gt = labels_gt[idx_pred_to_gt]
    # num_classes = labels_gt.max().item() + 1
    # print(num_classes)
    num_classes = len(class_names)
    print(num_classes)
    confmatrix = torch.zeros(num_classes, num_classes, device=labels_pred.device)
    for class_gt_int in range(num_classes):
        tensor_gt_class = torch.eq(labels_gt, class_gt_int).long()
        for class_pred_int in range(num_classes):
            tensor_pred_class = torch.eq(labels_pred, class_pred_int).long()
            tensor_pred_class = torch.mul(tensor_gt_class, tensor_pred_class)
            count = torch.sum(tensor_pred_class)
            confmatrix[class_gt_int, class_pred_int] += count

    return confmatrix


def compute_metrics(confmatrix, class_names):
    if isinstance(confmatrix, torch.Tensor):
        confmatrix = confmatrix.cpu().numpy()

    num_classes = len(class_names)
    ious = np.zeros((num_classes))
    precision = np.zeros((num_classes))
    recall = np.zeros((num_classes))
    f1score = np.zeros((num_classes))

    for _idx in range(num_classes):
        ious[_idx] = confmatrix[_idx, _idx] / (
            max(
                1,
                confmatrix[_idx, :].sum()
                + confmatrix[:, _idx].sum()
                - confmatrix[_idx, _idx],
            )
        )
        recall[_idx] = confmatrix[_idx, _idx] / max(1, confmatrix[_idx, :].sum())
        precision[_idx] = confmatrix[_idx, _idx] / max(1, confmatrix[:, _idx].sum())
        f1score[_idx] = (
            2 * precision[_idx] * recall[_idx] / max(1, precision[_idx] + recall[_idx])
        )

    fmiou = (ious * confmatrix.sum(1) / confmatrix.sum()).sum()
    # print(f"iou: {ious}")
    # print(f"miou: {ious.mean()}")
    # print(f"Acc>0.15: {(ious > 0.15).sum()}")
    # print(f"Acc>0.25: {(ious > 0.25).sum()}")
    # print(f"Acc>0.50: {(ious > 0.50).sum()}")
    # print(f"Acc>0.75: {(ious > 0.75).sum()}")
    # print(f"precision: {precision}")
    # print(f"recall: {recall}")
    # print(f"f1score: {f1score}")

    mdict = {}
    mdict["iou"] = ious.tolist()
    mdict["miou"] = ious.mean().item()
    mdict["fmiou"] = fmiou.item()
    mdict["num_classes"] = num_classes
    mdict["acc0.15"] = (ious > 0.15).sum().item()
    mdict["acc0.25"] = (ious > 0.25).sum().item()
    mdict["acc0.50"] = (ious > 0.50).sum().item()
    mdict["acc0.75"] = (ious > 0.75).sum().item()
    mdict["precision"] = precision.tolist()
    mdict["recall"] = recall.tolist()
    mdict["f1score"] = f1score.tolist()

    return mdict