from conceptgraph.utils.logging_metrics import MappingTracker
import torch
import torch.nn.functional as F

from typing import List, Optional

from conceptgraph.slam.slam_classes import MapObjectList, DetectionList
from conceptgraph.utils.general_utils import Timer
from conceptgraph.utils.ious import (
    compute_iou_batch, 
    compute_giou_batch, 
    compute_3d_iou_accurate_batch, 
    compute_3d_giou_accurate_batch,
)
from conceptgraph.slam.utils import (
    compute_overlap_matrix_general,
    merge_obj2_into_obj1, 
    compute_overlap_matrix_2set
)
from conceptgraph.utils.custom_wandb import OptionalWandB
optional_wandb = OptionalWandB()

tracker = MappingTracker()


def compute_spatial_similarities(spatial_sim_type: str, detection_list: DetectionList, objects: MapObjectList, downsample_voxel_size) -> torch.Tensor:
    det_bboxes = detection_list.get_stacked_values_torch('bbox')
    obj_bboxes = objects.get_stacked_values_torch('bbox')

    if spatial_sim_type == "iou":
        spatial_sim = compute_iou_batch(det_bboxes, obj_bboxes)
    elif spatial_sim_type == "giou":
        spatial_sim = compute_giou_batch(det_bboxes, obj_bboxes)
    elif spatial_sim_type == "iou_accurate":
        spatial_sim = compute_3d_iou_accurate_batch(det_bboxes, obj_bboxes)
    elif spatial_sim_type == "giou_accurate":
        spatial_sim = compute_3d_giou_accurate_batch(det_bboxes, obj_bboxes)
    elif spatial_sim_type == "overlap":
        spatial_sim = compute_overlap_matrix_general(objects, detection_list, downsample_voxel_size)
        spatial_sim = torch.from_numpy(spatial_sim).T
    else:
        raise ValueError(f"Invalid spatial similarity type: {spatial_sim_type}")
    
    return spatial_sim

def compute_visual_similarities(detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    Compute the visual similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of visual similarities
    '''
    det_fts = detection_list.get_stacked_values_torch('clip_ft') # (M, D)
    obj_fts = objects.get_stacked_values_torch('clip_ft') # (N, D)

    det_fts = det_fts.unsqueeze(-1) # (M, D, 1)
    obj_fts = obj_fts.T.unsqueeze(0) # (1, D, N)
    
    visual_sim = F.cosine_similarity(det_fts, obj_fts, dim=1) # (M, N)
    
    return visual_sim

def aggregate_similarities(match_method: str, phys_bias: float, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    Aggregate spatial and visual similarities into a single similarity score
    
    Args:
        spatial_sim: a MxN tensor of spatial similarities
        visual_sim: a MxN tensor of visual similarities
    Returns:
        A MxN tensor of aggregated similarities
    '''
    if match_method == "sim_sum":
        sims = (1 + phys_bias) * spatial_sim + (1 - phys_bias) * visual_sim
    else:
        raise ValueError(f"Unknown matching method: {match_method}")
    
    return sims

def match_detections_to_objects(
    agg_sim: torch.Tensor, detection_threshold: float = float('-inf')
) -> List[Optional[int]]:
    """
    Matches detections to objects based on similarity, returning match indices or None for unmatched.

    Args:
        agg_sim: Similarity matrix (detections vs. objects).
        detection_threshold: Threshold for a valid match (default: -inf).

    Returns:
        List of matching object indices (or None if unmatched) for each detection.
    """
    match_indices = []
    for detected_obj_idx in range(agg_sim.shape[0]):
        max_sim_value = agg_sim[detected_obj_idx].max()
        if max_sim_value <= detection_threshold:
            match_indices.append(None)
        else:
            match_indices.append(agg_sim[detected_obj_idx].argmax().item())

    return match_indices


def merge_obj_matches(
    detection_list: DetectionList,
    objects: MapObjectList,
    match_indices: List[Optional[int]],
    downsample_voxel_size: float,
    dbscan_remove_noise: bool,
    dbscan_eps: float,
    dbscan_min_points: int,
    spatial_sim_type: str,
    device: str,
) -> MapObjectList:
    """
    Merges detected objects into existing objects based on a list of match indices.

    Args:
        detection_list (DetectionList): List of detected objects.
        objects (MapObjectList): List of existing objects.
        match_indices (List[Optional[int]]): Indices of existing objects each detected object matches with.
        downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, spatial_sim_type, device:
            Parameters for merging and similarity computation.

    Returns:
        MapObjectList: Updated list of existing objects with detected objects merged as appropriate.
    """
    global tracker
    temp_curr_object_count = tracker.curr_object_count
    for detected_obj_idx, existing_obj_match_idx in enumerate(match_indices):
        if existing_obj_match_idx is None:
            # track the new object detection
            tracker.object_dict.update({
                "id": detection_list[detected_obj_idx]['id'],
                "first_discovered": tracker.curr_frame_idx
            })

            objects.append(detection_list[detected_obj_idx])
        else:

            detected_obj = detection_list[detected_obj_idx]
            matched_obj = objects[existing_obj_match_idx]
            merged_obj = merge_obj2_into_obj1(
                obj1=matched_obj,
                obj2=detected_obj,
                downsample_voxel_size=downsample_voxel_size,
                dbscan_remove_noise=dbscan_remove_noise,
                dbscan_eps=dbscan_eps,
                dbscan_min_points=dbscan_min_points,
                spatial_sim_type=spatial_sim_type,
                device=device,
                run_dbscan=False,
            )
            objects[existing_obj_match_idx] = merged_obj
    tracker.increment_total_merges(len(match_indices) - match_indices.count(None))
    tracker.increment_total_objects(len(objects) - temp_curr_object_count)
    # wandb.log({"merges_this_frame" :len(match_indices) - match_indices.count(None)})
    # wandb.log({"total_merges": tracker.total_merges})
    optional_wandb.log(
        {
            "merges_this_frame": len(match_indices) - match_indices.count(None),
            "total_merges": tracker.total_merges,
            "frame_idx": tracker.curr_frame_idx,
        }
    )
    return objects


def merge_detections_to_objects(
    downsample_voxel_size: float, dbscan_remove_noise: bool, dbscan_eps: float, dbscan_min_points: int,
    spatial_sim_type: str, device: str, match_method: str, phys_bias: float,
    detection_list: DetectionList, objects: MapObjectList, agg_sim: torch.Tensor
) -> MapObjectList:
    for detected_obj_idx in range(agg_sim.shape[0]):
        if agg_sim[detected_obj_idx].max() == float('-inf'):
            objects.append(detection_list[detected_obj_idx])
        else:
            existing_obj_match_idx = agg_sim[detected_obj_idx].argmax()
            detected_obj = detection_list[detected_obj_idx]
            matched_obj = objects[existing_obj_match_idx]
            merged_obj = merge_obj2_into_obj1(
                obj1=matched_obj, obj2=detected_obj, 
                downsample_voxel_size=downsample_voxel_size, dbscan_remove_noise=dbscan_remove_noise, 
                dbscan_eps=dbscan_eps, dbscan_min_points=dbscan_min_points, 
                spatial_sim_type=spatial_sim_type, device=device, 
                run_dbscan=False
            )
            objects[existing_obj_match_idx] = merged_obj
            
    return objects
