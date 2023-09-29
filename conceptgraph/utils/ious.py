import numpy as np
import torch

import open3d as o3d

def compute_3d_iou(bbox1, bbox2, padding=0, use_iou=True):
    # Get the coordinates of the first bounding box
    bbox1_min = np.asarray(bbox1.get_min_bound()) - padding
    bbox1_max = np.asarray(bbox1.get_max_bound()) + padding

    # Get the coordinates of the second bounding box
    bbox2_min = np.asarray(bbox2.get_min_bound()) - padding
    bbox2_max = np.asarray(bbox2.get_max_bound()) + padding

    # Compute the overlap between the two bounding boxes
    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)

    overlap_volume = np.prod(overlap_size)
    bbox1_volume = np.prod(bbox1_max - bbox1_min)
    bbox2_volume = np.prod(bbox2_max - bbox2_min)
    
    obj_1_overlap = overlap_volume / bbox1_volume
    obj_2_overlap = overlap_volume / bbox2_volume
    max_overlap = max(obj_1_overlap, obj_2_overlap)

    iou = overlap_volume / (bbox1_volume + bbox2_volume - overlap_volume)

    if use_iou:
        return iou
    else:
        return max_overlap

def compute_iou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    '''
    Compute IoU between two sets of axis-aligned 3D bounding boxes.
    
    bbox1: (M, V, D), e.g. (M, 8, 3)
    bbox2: (N, V, D), e.g. (N, 8, 3)
    
    returns: (M, N)
    '''
    # Compute min and max for each box
    bbox1_min, _ = bbox1.min(dim=1) # Shape: (M, 3)
    bbox1_max, _ = bbox1.max(dim=1) # Shape: (M, 3)
    bbox2_min, _ = bbox2.min(dim=1) # Shape: (N, 3)
    bbox2_max, _ = bbox2.max(dim=1) # Shape: (N, 3)

    # Expand dimensions for broadcasting
    bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, 3)
    bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, 3)

    # Compute max of min values and min of max values
    # to obtain the coordinates of intersection box.
    inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, 3)
    inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, 3)

    # Compute volume of intersection box
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # Shape: (M, N)

    # Compute volumes of the two sets of boxes
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)

    # Compute IoU, handling the special case where there is no intersection
    # by setting the intersection volume to 0.
    iou = inter_vol / (bbox1_vol + bbox2_vol - inter_vol + 1e-10)

    return iou
    
def compute_3d_giou(bbox1, bbox2):
    # Get the coordinates of the first bounding box
    bbox1_min = np.asarray(bbox1.get_min_bound())
    bbox1_max = np.asarray(bbox1.get_max_bound())

    # Get the coordinates of the second bounding box
    bbox2_min = np.asarray(bbox2.get_min_bound())
    bbox2_max = np.asarray(bbox2.get_max_bound())
    
    # Intersection
    intersec_min = np.maximum(bbox1_min, bbox2_min)
    intersec_max = np.minimum(bbox1_max, bbox2_max)
    intersec_size = np.maximum(intersec_max - intersec_min, 0.0)
    intersec_volume = np.prod(intersec_size)

    # Union
    bbox1_volume = np.prod(bbox1_max - bbox1_min)
    bbox2_volume = np.prod(bbox2_max - bbox2_min)
    union_volume = bbox1_volume + bbox2_volume - intersec_volume
    
    iou = intersec_volume / union_volume
    
    # Enclosing box
    enclosing_min = np.minimum(bbox1_min, bbox2_min)
    enclosing_max = np.maximum(bbox1_max, bbox2_max)
    enclosing_size = np.maximum(enclosing_max - enclosing_min, 0.0)
    enclosing_volume = np.prod(enclosing_size)
    
    giou = iou - (enclosing_volume - union_volume) / enclosing_volume
    
    return giou

def compute_giou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    '''
    Compute the generalized IoU between two sets of axis-aligned 3D bounding boxes.
    
    bbox1: (M, V, D), e.g. (M, 8, 3)
    bbox2: (N, V, D), e.g. (N, 8, 3)
    
    returns: (M, N)
    '''
    # Compute min and max for each box
    bbox1_min, _ = bbox1.min(dim=1) # Shape: (M, D)
    bbox1_max, _ = bbox1.max(dim=1) # Shape: (M, D)
    bbox2_min, _ = bbox2.min(dim=1) # Shape: (N, D)
    bbox2_max, _ = bbox2.max(dim=1) # Shape: (N, D)

    # Expand dimensions for broadcasting
    bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, D)
    bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, D)
    bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, D)
    bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, D)

    # Compute max of min values and min of max values
    # to obtain the coordinates of intersection box.
    inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, D)
    inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, D)
    
    # to obtain the coordinates of enclosing box
    enclosing_min = torch.min(bbox1_min, bbox2_min)  # Shape: (M, N, D)
    enclosing_max = torch.max(bbox1_max, bbox2_max)  # Shape: (M, N, D)

    # Compute volume of intersection box
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # Shape: (M, N)
    enclosing_vol = torch.prod(enclosing_max - enclosing_min, dim=2)  # Shape: (M, N)

    # Compute volumes of the two sets of boxes
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)
    union_vol = bbox1_vol + bbox2_vol - inter_vol

    # Compute IoU, handling the special case where there is no intersection
    # by setting the intersection volume to 0.
    iou = inter_vol / (union_vol + 1e-10)
    giou = iou - (enclosing_vol - union_vol) / (enclosing_vol + 1e-10)

    return giou

def compute_3d_iou_accuracte_batch(bbox1, bbox2):
    '''
    Compute IoU between two sets of oriented (or axis-aligned) 3D bounding boxes.
    
    bbox1: (M, 8, D), e.g. (M, 8, 3)
    bbox2: (N, 8, D), e.g. (N, 8, 3)
    
    returns: (M, N)
    '''
    # Must expend the box beforehand, otherwise it may results overestimated results
    bbox1 = expand_3d_box(bbox1, 0.02)
    bbox2 = expand_3d_box(bbox2, 0.02)
    
    import pytorch3d.ops as ops

    bbox1 = bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]]
    bbox2 = bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]]
    
    inter_vol, iou = ops.box3d_overlap(bbox1.float(), bbox2.float())
    
    return iou

def compute_3d_giou_accurate(obj1, obj2):
    '''
    Compute the 3D GIoU in a more accurate way. 
    '''
    import pytorch3d.ops as ops
    
    # This is too slow
    # bbox1 = pcd1.get_minimal_oriented_bounding_box()
    # bbox2 = pcd2.get_minimal_oriented_bounding_box()
    
    # This is still slow ... 
    # Moved it outside of this function so that it is computed less times
    # bbox1 = pcd1.get_oriented_bounding_box()
    # bbox2 = pcd2.get_oriented_bounding_box()
    
    bbox1 = obj1['bbox']
    bbox2 = obj2['bbox']
    pcd1 = obj1['pcd']
    pcd2 = obj2['pcd']
    
    # Get the coordinates of the bounding boxes
    box_points1 = np.asarray(bbox1.get_box_points())
    box_points2 = np.asarray(bbox2.get_box_points())
    
    # Re-order the points to fit the format required in pytorch3d
    # xyz should be [---, -+-, -++, --+,    +--, ++-, +++, +-+]
    box_points1 = box_points1[[0, 2, 5, 3, 1, 7, 4, 6]]
    box_points2 = box_points2[[0, 2, 5, 3, 1, 7, 4, 6]]
    
    # Computet the intersection of the two boxes
    try:
        vols, ious = ops.box3d_overlap(
            torch.from_numpy(box_points1).unsqueeze(0).float(), 
            torch.from_numpy(box_points2).unsqueeze(0).float()
        )
    except ValueError as e: # This indicates colinear
        union_volume = 0.0
        iou = 0.0
    else:
        union_volume = vols[0,0].item()
        iou = ious[0,0].item()
    
    # Join the two point cloud
    pcd_union = pcd1 + pcd2

    # compute_convex_hull() somehow cannot provide watertight mesh
    # union_hull_mesh, union_hull_point_list = pcd_union.compute_convex_hull()
    # enclosing_volume = union_hull_mesh.get_volume()
    
    # enclosing_box = pcd_union.get_minimal_oriented_bounding_box()
    enclosing_box = pcd_union.get_oriented_bounding_box()
    enclosing_volume = enclosing_box.volume()
    
    giou = iou - (enclosing_volume - union_volume) / enclosing_volume
    
    # print(vols, ious)
    # print(enclosing_volume, union_volume, iou, giou)
    
    # o3d.visualization.draw_geometries([
    #     pcd1, pcd2, bbox1, bbox2
    # ])
    
    return giou

def compute_3d_box_volume_batch(bbox: torch.Tensor) -> torch.Tensor:
    '''
    Compute the volume of a set of rectangular boxes.
    This assumes bbox corner order follows the open3d convention, which is:
    ---, +--, -+-, --+, +++, -++, +-+, ++-
    See https://github.com/isl-org/Open3D/blob/47f4ee936841ae9f8a9c4ce5d9162bd5b3e0279f/cpp/open3d/geometry/BoundingVolume.cpp#L92
    
    bbox: (N, 8, D)
    
    returns: (N,)
    '''
    a = torch.linalg.vector_norm(bbox[:, 0, :] - bbox[:, 1, :], ord=2, dim=1)
    b = torch.linalg.vector_norm(bbox[:, 0, :] - bbox[:, 2, :], ord=2, dim=1)
    c = torch.linalg.vector_norm(bbox[:, 0, :] - bbox[:, 3, :], ord=2, dim=1)
    
    vol = a * b * c
    return vol
    
def expand_3d_box(bbox: torch.Tensor, eps=0.02) -> torch.Tensor:
    '''
    Expand the side of 3D boxes such that each side has at least eps length.
    Assumes the bbox cornder order in open3d convention. 
    
    bbox: (N, 8, D)
    
    returns: (N, 8, D)
    '''
    center = bbox.mean(dim=1)  # shape: (N, D)

    va = bbox[:, 1, :] - bbox[:, 0, :]  # shape: (N, D)
    vb = bbox[:, 2, :] - bbox[:, 0, :]  # shape: (N, D)
    vc = bbox[:, 3, :] - bbox[:, 0, :]  # shape: (N, D)
    
    a = torch.linalg.vector_norm(va, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    b = torch.linalg.vector_norm(vb, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    c = torch.linalg.vector_norm(vc, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    
    va = torch.where(a < eps, va / a * eps, va)  # shape: (N, D)
    vb = torch.where(b < eps, vb / b * eps, vb)  # shape: (N, D)
    vc = torch.where(c < eps, vc / c * eps, vc)  # shape: (N, D)
    
    new_bbox = torch.stack([
        center - va/2.0 - vb/2.0 - vc/2.0,
        center + va/2.0 - vb/2.0 - vc/2.0,
        center - va/2.0 + vb/2.0 - vc/2.0,
        center - va/2.0 - vb/2.0 + vc/2.0,
        center + va/2.0 + vb/2.0 + vc/2.0,
        center - va/2.0 + vb/2.0 + vc/2.0,
        center + va/2.0 - vb/2.0 + vc/2.0,
        center + va/2.0 + vb/2.0 - vc/2.0,
    ], dim=1) # shape: (N, 8, D)
    
    new_bbox = new_bbox.to(bbox.device)
    new_bbox = new_bbox.type(bbox.dtype)
    
    return new_bbox
    
def compute_enclosing_vol(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    '''
    Compute the enclosing volume between every pair of boxes in bbox1 and bbox2.
    This is an accurate but slow version using convex hull
    
    bbox1: (M, 8, D)
    bbox2: (N, 8, D)
    
    returns: (M, N)
    '''
    M = bbox1.shape[0]
    N = bbox2.shape[0]
    
    enclosing_vol = torch.zeros((M, N), dtype=bbox1.dtype, device=bbox1.device)
    for i in range(bbox1.shape[0]):
        for j in range(bbox2.shape[0]):
            pcd_union = o3d.geometry.PointCloud()
            bbox_points_union = torch.cat([bbox1[i], bbox2[j]], dim=0) # (16, 3)
            pcd_union.points = o3d.utility.Vector3dVector(bbox_points_union.cpu().numpy())
            enclosing_mesh, _ = pcd_union.compute_convex_hull(joggle_inputs=True)
            try:
                enclosing_vol[i, j] = enclosing_mesh.get_volume()
            except:
                # This occurs commonly when the enclosing_mesh is not watertight.
                enclosing_mesh = pcd_union.get_axis_aligned_bounding_box()
                enclosing_vol[i, j] = enclosing_mesh.volume()
                
    return enclosing_vol
    
def compute_enclosing_vol_fast(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    '''
    Compute the enclosing volume between every pair of boxes in bbox1 and bbox2.
    This is fast but approximate version using axis-aligned bounding box
    
    bbox1: (M, 8, 3)
    bbox2: (N, 8, 3)
    
    returns: (M, N)
    '''
    M = bbox1.shape[0]
    N = bbox2.shape[0]
    
    # Expand dimensions to compute the pairwise enclosing box
    bbox1 = bbox1.unsqueeze(1).expand(-1, N, -1, -1) # (M, N, 8, 3)
    bbox2 = bbox2.unsqueeze(0).expand(M, -1, -1, -1) # (M, N, 8, 3)
    
    # Compute the minimum and maximum coordinates for each pair of boxes
    min_coords = torch.minimum(bbox1, bbox2).amin(dim=2) # (M, N, 3)
    max_coords = torch.maximum(bbox1, bbox2).amax(dim=2) # (M, N, 3)

    # Compute the dimensions of the enclosing boxes
    enclosing_dims = max_coords - min_coords # (M, N, 3)
    
    # Clamp the dimensions to be non-negative (in case there's no overlap)
    enclosing_dims = torch.clamp(enclosing_dims, min=0) # (M, N, 3)
    
    # Compute the volume of the enclosing boxes
    vol = enclosing_dims[:, :, 0] * enclosing_dims[:, :, 1] * enclosing_dims[:, :, 2] # (M, N)

    return vol

def compute_3d_giou_accurate_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    '''
    Compute Generalized IoU between two sets of oriented (or axis-aligned) 3D bounding boxes.
    
    bbox1: (M, 8, D), e.g. (M, 8, 3)
    bbox2: (N, 8, D), e.g. (N, 8, 3)
    
    returns: (M, N)
    '''
    # Must expend the box beforehand, otherwise it may results overestimated results
    bbox1 = expand_3d_box(bbox1, 0.02)
    bbox2 = expand_3d_box(bbox2, 0.02)
    
    bbox1_vol = compute_3d_box_volume_batch(bbox1)
    bbox2_vol = compute_3d_box_volume_batch(bbox2)
    
    import pytorch3d.ops as ops

    inter_vol, iou = ops.box3d_overlap(
        bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]].float(), 
        bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]].float()
    )
    union_vol = bbox1_vol.unsqueeze(1) + bbox2_vol.unsqueeze(0) - inter_vol
    
    enclosing_vol = compute_enclosing_vol(bbox1, bbox2)
    # enclosing_vol = compute_enclosing_vol_fast(bbox1, bbox2)
    
    giou = iou - (enclosing_vol - union_vol) / enclosing_vol
    
    return giou

def compute_3d_contain_ratio_accurate_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    '''
    Compute for i-th box in bbox1, how much of it is contained in j-th box in bbox2.
    
    bbox1: (M, 8, D), e.g. (M, 8, 3)
    bbox2: (N, 8, D), e.g. (N, 8, 3)
    
    returns: (M, N)
    '''
    # Must expend the box beforehand, otherwise it may results overestimated results
    bbox1 = expand_3d_box(bbox1)
    bbox2 = expand_3d_box(bbox2)
    
    bbox1_vol = compute_3d_box_volume_batch(bbox1) # (M,)
    bbox2_vol = compute_3d_box_volume_batch(bbox2) # (M,)
    
    import pytorch3d.ops as ops
    
    inter_vol, iou = ops.box3d_overlap(
        bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]].float(), 
        bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]].float()
    ) # (M, N), (M, N)
    
    contain_ratio = inter_vol / bbox1_vol.unsqueeze(1) # (M, N)
    
    # Seems the following bug is unavoidable but happens very rarely
    # print((contain_ratio > 1.001).sum() / contain_ratio.numel())
    # if contain_ratio.amax() > 1.1:
    #     print('contain_ratio > 1.0')
    #     import pdb; pdb.set_trace()
    
    # Therefore we manually clamp it to [0, 1]
    contain_ratio = contain_ratio.clamp(min=0, max=1)
    
    return contain_ratio, iou

def compute_2d_box_contained_batch(bbox: torch.Tensor, thresh:float=0.95) -> torch.Tensor:
    '''
    For each bbox, compute how many other bboxes are containing it. 
    First compute the area of the intersection between each pair of bboxes. 
    Then for each bbox, count how many bboxes have the intersection area larger than thresh of its own area.
    
    bbox: (N, 4), in (x1, y1, x2, y2) format
    
    returns: (N,)
    '''
    N = bbox.shape[0]

    # Get areas of each bbox
    areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

    # Compute intersection boxes
    lt = torch.max(bbox[:, None, :2], bbox[:, :2])  # left-top points
    rb = torch.min(bbox[:, None, 2:], bbox[:, 2:])  # right-bottom points

    inter = (rb - lt).clamp(min=0)  # intersection sizes (dx, dy), if no overlap, clamp to zero

    # Compute areas of intersection boxes
    inter_areas = inter[:, :, 0] * inter[:, :, 1]

    # Count how many boxes have intersection area larger than thresh of its own area
    mask = inter_areas > (areas * thresh).unsqueeze(1)
    count = mask.sum(dim=1) - 1  # exclude itself

    return count

def mask_subtract_contained(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    '''
    Compute the containing relationship between all pair of bounding boxes.
    For each mask, subtract the mask of bounding boxes that are contained by it.
     
    Args:
        xyxy: (N, 4), in (x1, y1, x2, y2) format
        mask: (N, H, W), binary mask
        th1: float, threshold for computing intersection over box1
        th2: float, threshold for computing intersection over box2
        
    Returns:
        mask_sub: (N, H, W), binary mask
    '''
    N = xyxy.shape[0] # number of boxes

    # Get areas of each xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) # (N,)

    # Compute intersection boxes
    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
    rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])  # right-bottom points (N, N, 2)
    
    inter = (rb - lt).clip(min=0)  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

    # Compute areas of intersection boxes
    inter_areas = inter[:, :, 0] * inter[:, :, 1] # (N, N)
    
    inter_over_box1 = inter_areas / areas[:, None] # (N, N)
    # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
    inter_over_box2 = inter_over_box1.T # (N, N)
    
    # if the intersection area is smaller than th2 of the area of box1, 
    # and the intersection area is larger than th1 of the area of box2,
    # then box2 is considered contained by box1
    contained = (inter_over_box1 < th2) & (inter_over_box2 > th1) # (N, N)
    contained_idx = contained.nonzero() # (num_contained, 2)

    mask_sub = mask.copy() # (N, H, W)
    # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
    for i in range(len(contained_idx[0])):
        mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (~mask_sub[contained_idx[1][i]])

    return mask_sub