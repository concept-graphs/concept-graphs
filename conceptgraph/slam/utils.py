from collections import Counter
import copy
import json
import logging
from pathlib import Path
# from conceptgraph.utils.logging_metrics import track_denoising, 
from conceptgraph.utils.logging_metrics import DenoisingTracker, MappingTracker 
import cv2
# from line_profiler import profile

import numpy as np
from omegaconf import DictConfig
import omegaconf
import open3d as o3d
import torch

import torch.nn.functional as F

import faiss
import uuid

from conceptgraph.utils.general_utils import measure_time, to_tensor, to_numpy, Timer
from conceptgraph.slam.slam_classes import MapObjectList, DetectionList

from conceptgraph.utils.ious import compute_3d_iou, compute_3d_iou_accurate_batch, compute_iou_batch
from conceptgraph.dataset.datasets_common import from_intrinsics_matrix

tracker = MappingTracker()

def get_classes_colors(classes):
    class_colors = {}

    # Generate a random color for each class
    for class_idx, class_name in enumerate(classes):
        # Generate random RGB values between 0 and 255
        r = np.random.randint(0, 256)/255.0
        g = np.random.randint(0, 256)/255.0
        b = np.random.randint(0, 256)/255.0

        # Assign the RGB values as a tuple to the class in the dictionary
        class_colors[class_idx] = (r, g, b)

    class_colors[-1] = (0, 0, 0)

    return class_colors

def create_or_load_colors(cfg, filename="gsa_classes_tag2text"):
    
    # get the classes, should be saved when making the dataset
    classes_fp = cfg['dataset_root'] / cfg['scene_id'] / f"{filename}.json"
    classes  = None
    with open(classes_fp, "r") as f:
        classes = json.load(f)
    
    # create the class colors, or load them if they exist
    class_colors  = None
    class_colors_fp = cfg['dataset_root'] / cfg['scene_id'] / f"{filename}_colors.json"
    if class_colors_fp.exists():
        with open(class_colors_fp, "r") as f:
            class_colors = json.load(f)
        print("Loaded class colors from ", class_colors_fp)
    else:
        class_colors = get_classes_colors(classes)
        class_colors = {str(k): v for k, v in class_colors.items()}
        with open(class_colors_fp, "w") as f:
            json.dump(class_colors, f)
        print("Saved class colors to ", class_colors_fp)
    return classes, class_colors

# @profile
def create_object_pcd(depth_array, mask, cam_K, image, obj_color=None) -> o3d.geometry.PointCloud:
    fx, fy, cx, cy = from_intrinsics_matrix(cam_K)
    
    # Also remove points with invalid depth values
    mask = np.logical_and(mask, depth_array > 0)

    # if no valid points, return an empty point cloud
    if not np.any(mask):
        pcd = o3d.geometry.PointCloud()
        return pcd
        
    height, width = depth_array.shape
    x = np.arange(0, width, 1.0)
    y = np.arange(0, height, 1.0)
    u, v = np.meshgrid(x, y)
    
    # Apply the mask, and unprojection is done only on the valid points
    masked_depth = depth_array[mask] # (N, )
    u = u[mask] # (N, )
    v = v[mask] # (N, )

    # Convert to 3D coordinates
    x = (u - cx) * masked_depth / fx
    y = (v - cy) * masked_depth / fy
    z = masked_depth

    # Stack x, y, z coordinates into a 3D point cloud
    points = np.stack((x, y, z), axis=-1)
    points = points.reshape(-1, 3)
    
    # Perturb the points a bit to avoid colinearity
    points += np.random.normal(0, 4e-3, points.shape)

    if obj_color is None: # color using RGB
        # # Apply mask to image
        colors = image[mask] / 255.0
    else: # color using group ID
        # Use the assigned obj_color for all points
        colors = np.full(points.shape, obj_color)
    
    if points.shape[0] == 0:
        import pdb; pdb.set_trace()

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

# @profile
def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    ## Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )
    
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd
        
    return pcd

def init_pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    ## Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan( # inint
        eps=eps,
        min_points=min_points,
    )
    
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd
        
    return pcd

def init_process_pcd(pcd, downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, run_dbscan=True):
    pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
    
    if dbscan_remove_noise and run_dbscan:
        pcd = init_pcd_denoise_dbscan(
            pcd, 
            eps=dbscan_eps, 
            min_points=dbscan_min_points
        )
        
    return pcd

# @profile
def process_pcd(pcd, downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, run_dbscan=True):
    pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
    
    if dbscan_remove_noise and run_dbscan:
        pass
        # pcd = pcd_denoise_dbscan(
        #     pcd, 
        #     eps=dbscan_eps, 
        #     min_points=dbscan_min_points
        # )
        
    return pcd

# @profile
def get_bounding_box(spatial_sim_type, pcd):
    if ("accurate" in spatial_sim_type or "overlap" in spatial_sim_type) and len(pcd.points) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()
    else:
        return pcd.get_axis_aligned_bounding_box()

# @profile
def merge_obj2_into_obj1(obj1, obj2, downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, spatial_sim_type, device, run_dbscan=True):

    '''
    Merge the new object to the old object
    This operation is done in-place
    '''
    global tracker
    
    tracker.track_merge(obj1, obj2)
    
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    
    for k in obj1.keys():
        if k in ['class_name', 'mask', 'id', 'curr_obj_num','new_counter', 'num_obj_in_class']:
            continue
        if k in ['caption']:
            # Here we need to merge two dictionaries and adjust the key of the second one
            for k2, v2 in obj2['caption'].items():
                obj1['caption'][k2 + n_obj1_det] = v2
        elif k not in ['pcd', 'bbox', 'clip_ft', "text_ft", "class_name"]:
            if isinstance(obj1[k], list) or isinstance(obj1[k], int):
                obj1[k] += obj2[k]
            elif k == "inst_color":
                obj1[k] = obj1[k] # Keep the initial instance color
            else:
                # TODO: handle other types if needed in the future
                raise NotImplementedError
        else: # pcd, bbox, clip_ft, text_ft are handled below
            continue

    # merge pcd and bbox
    obj1['pcd'] += obj2['pcd']
    obj1['pcd'] = process_pcd(obj1['pcd'], downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, run_dbscan)
    obj1['bbox'] = get_bounding_box(spatial_sim_type, obj1['pcd'])
    obj1['bbox'].color = [0,1,0]
    
    # merge clip ft
    obj1['clip_ft'] = (obj1['clip_ft'] * n_obj1_det +
                       obj2['clip_ft'] * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['clip_ft'] = F.normalize(obj1['clip_ft'], dim=0)

    # merge text_ft
    obj2['text_ft'] = to_tensor(obj2['text_ft'], device)
    obj1['text_ft'] = to_tensor(obj1['text_ft'], device)
    obj1['text_ft'] = (obj1['text_ft'] * n_obj1_det +
                       obj2['text_ft'] * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['text_ft'] = F.normalize(obj1['text_ft'], dim=0)
    
    return obj1

def compute_overlap_matrix(objects: MapObjectList, downsample_voxel_size):
    '''
    compute pairwise overlapping between objects in terms of point nearest neighbor. 
    Suppose we have a list of n point cloud, each of which is a o3d.geometry.PointCloud object. 
    Now we want to construct a matrix of size n x n, where the (i, j) entry is the ratio of points in point cloud i 
    that are within a distance threshold of any point in point cloud j. 
    '''
    n = len(objects)
    overlap_matrix = np.zeros((n, n))
    
    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    point_arrays = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects]
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in point_arrays]
    
    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, point_arrays):
        index.add(arr)

    # Compute the pairwise overlaps
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip diagonal elements
                box_i = objects[i]['bbox']
                box_j = objects[j]['bbox']
                
                # Skip if the boxes do not overlap at all (saves computation)
                iou = compute_3d_iou(box_i, box_j)
                if iou == 0:
                    continue
                
                # # Use range_search to find points within the threshold
                # _, I = indices[j].range_search(point_arrays[i], threshold ** 2)
                D, I = indices[j].search(point_arrays[i], 1)

                # # If any points are found within the threshold, increase overlap count
                # overlap += sum([len(i) for i in I])

                overlap = (D < downsample_voxel_size ** 2).sum() # D is the squared distance

                # Calculate the ratio of points within the threshold
                overlap_matrix[i, j] = overlap / len(point_arrays[i])

    return overlap_matrix

def compute_overlap_matrix_2set(objects_map: MapObjectList, objects_new: DetectionList, downsample_voxel_size) -> np.ndarray:
    """
    Computes pairwise overlap between two sets of objects based on point proximity. 
    This function evaluates how much each new object overlaps with each existing object in the map by calculating the ratio of points in one object's point cloud that are within a specified distance threshold of points in the other object's point cloud.

    Args:
        objects_map (MapObjectList): The existing objects in the map, where each object includes a point cloud.
        objects_new (DetectionList): The new objects to be added to the map, each with its own point cloud.
        downsample_voxel_size (float): The distance threshold for considering points as overlapping. Points within this distance are counted as overlapping.

    Returns:
        np.ndarray: An overlap matrix of size m x n, where m is the number of existing objects and n is the number of new objects. Each entry (i, j) in the matrix represents the ratio of points in the i-th existing object's point cloud that are within the distance threshold of any point in the j-th new object's point cloud.

    Note:
        - The overlap matrix helps identify potential duplicates or matches between new and existing objects based on spatial overlap.
        - High values (e.g., >0.8) in the matrix suggest a significant overlap, potentially indicating duplicates or very close matches.
        - Moderate values (e.g., 0.5-0.8) may indicate similar objects with partial overlap.
        - Low values (<0.5) generally suggest distinct objects with minimal overlap.
        - The choice of a "match" threshold depends on the application's requirements and may require adjusting based on observed outcomes.
    """
    m = len(objects_map)
    n = len(objects_new)
    overlap_matrix = np.zeros((m, n))
    
    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    points_map = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_map] # m arrays
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in points_map] # m indices
    
    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, points_map):
        index.add(arr)
        
    points_new = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_new] # n arrays
        
    bbox_map = objects_map.get_stacked_values_torch('bbox')
    bbox_new = objects_new.get_stacked_values_torch('bbox')
    try:
        iou = compute_3d_iou_accurate_batch(bbox_map, bbox_new) # (m, n)
    except ValueError:
        print("Met `Plane vertices are not coplanar` error, use axis aligned bounding box instead")
        bbox_map = []
        bbox_new = []
        for pcd in objects_map.get_values('pcd'):
            bbox_map.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        for pcd in objects_new.get_values('pcd'):
            bbox_new.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        bbox_map = torch.from_numpy(np.stack(bbox_map))
        bbox_new = torch.from_numpy(np.stack(bbox_new))
        
        iou = compute_iou_batch(bbox_map, bbox_new) # (m, n)
            

    # Compute the pairwise overlaps
    for i in range(m):
        for j in range(n):
            if iou[i,j] < 1e-6:
                continue
            
            D, I = indices[i].search(points_new[j], 1) # search new object j in map object i

            overlap = (D < downsample_voxel_size ** 2).sum() # D is the squared distance

            # Calculate the ratio of points within the threshold
            overlap_matrix[i, j] = overlap / len(points_new[j])

    return overlap_matrix

# @profile
def compute_overlap_matrix_general(objects_a: MapObjectList, objects_b = None, downsample_voxel_size = None) -> np.ndarray:
    """
    Compute the overlap matrix between two sets of objects represented by their point clouds. This function can also perform self-comparison when `objects_b` is not provided. The overlap is quantified based on the proximity of points from one object to the nearest points of another, within a threshold specified by `downsample_voxel_size`.

    Parameters
    ----------
    objects_a : MapObjectList
        A list of object representations where each object contains a point cloud ('pcd') and bounding box ('bbox').
        This is the primary set of objects for comparison.

    objects_b : Optional[MapObjectList]
        A second list of object representations similar to `objects_a`. If None, `objects_a` will be compared with itself to calculate self-overlap. Defaults to None.

    downsample_voxel_size : Optional[float]
        The threshold for determining whether points are close enough to be considered overlapping. Specifically, it's the square of the maximum distance allowed between points from two objects to consider those points as overlapping.
        Must be provided; if None, a ValueError is raised.

    Returns
    -------
    np.ndarray
        A 2D numpy array of shape (len(objects_a), len(objects_b)) containing the overlap ratios between objects.
        The overlap ratio is defined as the fraction of points in the second object's point cloud that are within `downsample_voxel_size` distance to any point in the first object's point cloud.

    Raises
    ------
    ValueError
        If `downsample_voxel_size` is not provided.

    Notes
    -----
    The function uses the FAISS library for efficient nearest neighbor searches to compute the overlap.
    Additionally, it employs a 3D IoU (Intersection over Union) computation for bounding boxes to quickly filter out pairs of objects without spatial overlap, improving performance.
    - The overlap matrix helps identify potential duplicates or matches between new and existing objects based on spatial overlap.
    - High values (e.g., >0.8) in the matrix suggest a significant overlap, potentially indicating duplicates or very close matches.
    - Moderate values (e.g., 0.5-0.8) may indicate similar objects with partial overlap.
    - Low values (<0.5) generally suggest distinct objects with minimal overlap.
    - The choice of a "match" threshold depends on the application's requirements and may require adjusting based on observed outcomes.

    Examples
    --------
    >>> objects_a = [{'pcd': pcd1, 'bbox': bbox1}, {'pcd': pcd2, 'bbox': bbox2}]
    >>> objects_b = [{'pcd': pcd3, 'bbox': bbox3}, {'pcd': pcd4, 'bbox': bbox4}]
    >>> downsample_voxel_size = 0.05
    >>> overlap_matrix = compute_overlap_matrix_general(objects_a, objects_b, downsample_voxel_size)
    >>> print(overlap_matrix)
    """
    # if downsample_voxel_size is None, raise an error
    if downsample_voxel_size is None:
        raise ValueError("downsample_voxel_size is not provided")

    # hardcoding for now because its this value is actually not supposed to be the downsample voxel size
    downsample_voxel_size = 0.025

    # are we doing self comparison?
    same_objects = objects_b is None
    objects_b = objects_a if same_objects else objects_b

    len_a = len(objects_a)
    len_b = len(objects_b)
    overlap_matrix = np.zeros((len_a, len_b))

    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    points_a = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_a] # m arrays
    indices_a = [faiss.IndexFlatL2(points_a_arr.shape[1]) for points_a_arr in points_a] # m indices

    # Add the points from the numpy arrays to the corresponding FAISS indices
    for idx_a, points_a_arr in zip(indices_a, points_a):
        idx_a.add(points_a_arr)

    points_b = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_b] # n arrays

    bbox_a = objects_a.get_stacked_values_torch('bbox')
    bbox_b = objects_b.get_stacked_values_torch('bbox')
    ious = compute_3d_iou_accurate_batch(bbox_a, bbox_b) # (m, n)


    # Compute the pairwise overlaps
    for idx_a in range(len_a):
        for idx_b in range(len_b):

            # skip same object comparison if same_objects is True
            if same_objects and idx_a == idx_b:
                continue

            # skip if the boxes do not overlap at all
            if ious[idx_a,idx_b] < 1e-6:
                continue

            # get the distance of the nearest neighbor of
            # each point in points_b[idx_b] to the points_a[idx_a]
            D, I = indices_a[idx_a].search(points_b[idx_b], 1) 
            overlap = (D < downsample_voxel_size ** 2).sum() # D is the squared distance

            # Calculate the ratio of points within the threshold
            overlap_matrix[idx_a, idx_b] = overlap / len(points_b[idx_b])

    return overlap_matrix

# @profile
def merge_overlap_objects(
    merge_overlap_thresh: float,
    merge_visual_sim_thresh: float,
    merge_text_sim_thresh: float,
    objects: MapObjectList,
    overlap_matrix: np.ndarray,
    downsample_voxel_size: float,
    dbscan_remove_noise: bool,
    dbscan_eps: float,
    dbscan_min_points: int,
    spatial_sim_type: str,
    device: str,
):
    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]
    
    # Sort indices of overlap ratios in descending order
    sort = np.argsort(overlap_ratio)[::-1]  
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]

    kept_objects = np.ones(
        len(objects), dtype=bool
    )  # Initialize all objects as 'kept' initially
    for i, j, ratio in zip(x, y, overlap_ratio):
        if ratio > merge_overlap_thresh:
            visual_sim = F.cosine_similarity(
                to_tensor(objects[i]["clip_ft"]),
                to_tensor(objects[j]["clip_ft"]),
                dim=0,
            )
            text_sim = F.cosine_similarity(
                to_tensor(objects[i]["text_ft"]),
                to_tensor(objects[j]["text_ft"]),
                dim=0,
            )
            if (
                visual_sim > merge_visual_sim_thresh
                and text_sim > merge_text_sim_thresh
            ):
                if kept_objects[
                    j
                ]:  # Check if the target object has not been merged into another
                    # Merge object i into object j
                    objects[j] = merge_obj2_into_obj1(
                        objects[j],
                        objects[i],
                        downsample_voxel_size,
                        dbscan_remove_noise,
                        dbscan_eps,
                        dbscan_min_points,
                        spatial_sim_type,
                        device,
                        run_dbscan=True,
                    )
                    kept_objects[i] = False  # Mark object i as 'merged'
        else:
            break  # Stop processing if the current overlap ratio is below the threshold

    # Create a new list of objects excluding those that were merged
    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)

    return objects

# @profile
def denoise_objects(
    downsample_voxel_size: float,
    dbscan_remove_noise: bool,
    dbscan_eps: float,
    dbscan_min_points: int,
    spatial_sim_type: str,
    device: str,
    objects: MapObjectList,
):
    tracker = DenoisingTracker()  # Get the singleton instance of DenoisingTracker
    logging.debug(f"Starting denoising with {len(objects)} objects")
    for i in range(len(objects)):
        og_object_pcd = objects[i]["pcd"]
        
        if len(og_object_pcd.points) <= 1: # no need to denoise
            objects[i]["pcd"] = og_object_pcd
        else:
            # Adjust the call to process_pcd with explicit parameters
            objects[i]["pcd"] = process_pcd(
                objects[i]["pcd"],
                downsample_voxel_size,
                dbscan_remove_noise,
                dbscan_eps,
                dbscan_min_points,
                run_dbscan=True,
            )
            if len(objects[i]["pcd"].points) < 4:
                objects[i]["pcd"] = og_object_pcd

        # Adjust the call to get_bounding_box with explicit parameters
        objects[i]["bbox"] = get_bounding_box(spatial_sim_type, objects[i]["pcd"])
        objects[i]["bbox"].color = [0, 1, 0]
        logging.debug(f"Finished denoising object {i} out of {len(objects)}")
        # Use the tracker's method
        tracker.track_denoising(objects[i]["id"], len(og_object_pcd.points), len(objects[i]["pcd"].points))
        
        # track_denoising(objects[i]["id"], len(og_object_pcd.points), len(objects[i]["pcd"].points))
        logging.debug(f"before denoising: {len(og_object_pcd.points)}, after denoising: {len(objects[i]['pcd'].points)}")
    logging.debug(f"Finished denoising with {len(objects)} objects")
    return objects

# @profile
def filter_objects(
    obj_min_points: int, obj_min_detections: int, objects: MapObjectList
):
    print("Before filtering:", len(objects))
    objects_to_keep = []
    for obj in objects:
        if (
            len(obj["pcd"].points) >= obj_min_points
            and obj["num_detections"] >= obj_min_detections
        ):
            objects_to_keep.append(obj)
    objects = MapObjectList(objects_to_keep)
    print("After filtering:", len(objects))

    return objects

# @profile
def merge_objects(
    merge_overlap_thresh: float,
    merge_visual_sim_thresh: float,
    merge_text_sim_thresh: float,
    objects: MapObjectList,
    downsample_voxel_size: float,
    dbscan_remove_noise: bool,
    dbscan_eps: float,
    dbscan_min_points: int,
    spatial_sim_type: str,
    device: str,
):
    if len(objects) == 0:
        return objects
    if merge_overlap_thresh <= 0:
        return objects

    # Assuming compute_overlap_matrix requires only `objects` and `downsample_voxel_size`
    overlap_matrix = compute_overlap_matrix_general(
        objects_a=objects,
        objects_b=None,
        downsample_voxel_size=downsample_voxel_size,
    )
    print("Before merging:", len(objects))
    # Pass all necessary configuration parameters to merge_overlap_objects
    objects = merge_overlap_objects(
        merge_overlap_thresh=merge_overlap_thresh,
        merge_visual_sim_thresh=merge_visual_sim_thresh,
        merge_text_sim_thresh=merge_text_sim_thresh,
        objects=objects,
        overlap_matrix=overlap_matrix,
        downsample_voxel_size=downsample_voxel_size,
        dbscan_remove_noise=dbscan_remove_noise,
        dbscan_eps=dbscan_eps,
        dbscan_min_points=dbscan_min_points,
        spatial_sim_type=spatial_sim_type,
        device=device,
    )
    print("After merging:", len(objects))

    return objects


# @profile
def filter_gobs(
    gobs: dict,
    image: np.ndarray,
    skip_bg: bool = None,  # Explicitly passing skip_bg
    BG_CLASSES: list = None,  # Explicitly passing BG_CLASSES
    mask_area_threshold: float = 10,  # Default value as fallback
    max_bbox_area_ratio: float = None,  # Explicitly passing max_bbox_area_ratio
    mask_conf_threshold: float = None,  # Explicitly passing mask_conf_threshold
):
    # If no detection at all
    if len(gobs['xyxy']) == 0:
        return gobs

    # Filter out the objects based on various criteria
    idx_to_keep = []
    for mask_idx in range(len(gobs['xyxy'])):
        local_class_id = gobs['class_id'][mask_idx]
        class_name = gobs['classes'][local_class_id]

        # Skip masks that are too small
        mask_area = gobs['mask'][mask_idx].sum()
        if mask_area < max(mask_area_threshold, 10):
            logging.debug(f"Skipped due to small mask area ({mask_area} pixels) - Class: {class_name}")
            continue

        # Skip the BG classes
        if skip_bg and class_name in BG_CLASSES:
            logging.debug(f"Skipped background class: {class_name}")
            continue

        # Skip the non-background boxes that are too large
        if class_name not in BG_CLASSES:
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            if max_bbox_area_ratio is not None and bbox_area > max_bbox_area_ratio * image_area:
                logging.debug(f"Skipped due to large bounding box area ratio - Class: {class_name}, Area Ratio: {bbox_area/image_area:.4f}")
                continue

        # Skip masks with low confidence
        if mask_conf_threshold is not None and gobs['confidence'] is not None:
            if gobs['confidence'][mask_idx] < mask_conf_threshold:
                # logging.debug(f"Skipped due to low confidence ({gobs['confidence'][mask_idx]}) - Class: {class_name}")
                continue

        idx_to_keep.append(mask_idx)

    # for key in gobs.keys():
    #     print(key, type(gobs[key]), len(gobs[key]))

    for k in gobs.keys():
        if isinstance(gobs[k], str) or k == "classes":  # Captions
            continue
        elif isinstance(gobs[k], list):
            gobs[k] = [gobs[k][i] for i in idx_to_keep]
        elif isinstance(gobs[k], np.ndarray):
            gobs[k] = gobs[k][idx_to_keep]
        else:
            raise NotImplementedError(f"Unhandled type {type(gobs[k])}")

    return gobs


def resize_gobs(gobs, image):

    # If the shapes are the same, no resizing is necessary
    if gobs['mask'].shape[1:] == image.shape[:2]:
        return gobs

    new_masks = []

    for mask_idx in range(len(gobs['xyxy'])):
        # TODO: rewrite using interpolation/resize in numpy or torch rather than cv2
        mask = gobs['mask'][mask_idx]
        # Rescale the xyxy coordinates to the image shape
        x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
        x1 = round(x1 * image.shape[1] / mask.shape[1])
        y1 = round(y1 * image.shape[0] / mask.shape[0])
        x2 = round(x2 * image.shape[1] / mask.shape[1])
        y2 = round(y2 * image.shape[0] / mask.shape[0])
        gobs['xyxy'][mask_idx] = [x1, y1, x2, y2]

        # Reshape the mask to the image shape
        mask = cv2.resize(mask.astype(np.uint8), image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
        new_masks.append(mask)

    if len(new_masks) > 0:
        gobs['mask'] = np.asarray(new_masks)

    return gobs

# # @profile
# def gobs_to_detection_list(
#     image, 
#     depth_array,
#     cam_K, 
#     idx, 
#     gobs, 
#     trans_pose = None,
#     class_names = None,
#     BG_CLASSES  = None,
#     color_path = None,
#     min_points_threshold: int = None,
#     spatial_sim_type: str = None,
#     downsample_voxel_size: float = None,  # New parameter
#     dbscan_remove_noise: bool = None,     # New parameter
#     dbscan_eps: float = None,             # New parameter
#     dbscan_min_points: int = None         # New parameter
# ):
#     '''
#     Return a DetectionList object from the gobs
#     All object are still in the camera frame. 
#     '''
#     fg_detection_list = DetectionList()
#     bg_detection_list = DetectionList()
    
#     # gobs = resize_gobs(gobs, image)
#     # gobs = filter_gobs(
#     #     gobs, 
#     #     image, 
#     #     skip_bg=skip_bg,
#     #     BG_CLASSES=BG_CLASSES,
#     #     mask_area_threshold=mask_area_threshold,
#     #     max_bbox_area_ratio=max_bbox_area_ratio,
#     #     mask_conf_threshold=mask_conf_threshold,
#     # )

#     if len(gobs['xyxy']) == 0:
#         return fg_detection_list, bg_detection_list
    
#     # Compute the containing relationship among all detections and subtract fg from bg objects
#     xyxy = gobs['xyxy']
#     mask = gobs['mask']
#     # gobs['mask'] = mask_subtract_contained(xyxy, mask)
    
#     n_masks = len(gobs['xyxy'])
#     for mask_idx in range(n_masks):
#         local_class_id = gobs['class_id'][mask_idx]
#         mask = gobs['mask'][mask_idx]
#         class_name = gobs['classes'][local_class_id]
#         global_class_id = -1 if class_names is None else class_names.index(class_name)
        
#         # make the pcd and color it
#         camera_object_pcd = create_object_pcd(
#             depth_array,
#             mask,
#             cam_K,
#             image,
#             obj_color = None
#         )
        
#         # It at least contains 5 points
#         if len(camera_object_pcd.points) < max(min_points_threshold, 5): 
#             continue
        
#         if trans_pose is not None:
#             global_object_pcd = camera_object_pcd.transform(trans_pose)
#         else:
#             global_object_pcd = camera_object_pcd
        
#         # get largest cluster, filter out noise 
#         # global_object_pcd = process_pcd(global_object_pcd, cfg)
#         global_object_pcd = process_pcd(global_object_pcd, downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, run_dbscan=True)

        
#         # pcd_bbox = get_bounding_box(cfg, global_object_pcd)
#         pcd_bbox = get_bounding_box(spatial_sim_type, global_object_pcd)
#         pcd_bbox.color = [0,1,0]
        
#         if pcd_bbox.volume() < 1e-6:
#             continue
        
#         # Treat the detection in the same way as a 3D object
#         # Store information that is enough to recover the detection
#         detected_object = {
#             'id' : uuid.uuid4(),
#             'image_idx' : [idx],                             # idx of the image
#             'mask_idx' : [mask_idx],                         # idx of the mask/detection
#             'color_path' : [color_path],                     # path to the RGB image
#             'class_name' : [class_name],                         # global class id for this detection
#             'class_id' : [global_class_id],                         # global class id for this detection
#             'num_detections' : 1,                            # number of detections in this object
#             'mask': [mask],
#             'xyxy': [gobs['xyxy'][mask_idx]],
#             'conf': [gobs['confidence'][mask_idx]],
#             'n_points': [len(global_object_pcd.points)],
#             'pixel_area': [mask.sum()],
#             'contain_number': [None],                          # This will be computed later
#             "inst_color": np.random.rand(3),                 # A random color used for this segment instance
#             'is_background': class_name in BG_CLASSES,
            
#             # These are for the entire 3D object
#             'pcd': global_object_pcd,
#             'bbox': pcd_bbox,
#             'clip_ft': to_tensor(gobs['image_feats'][mask_idx]),
#             'text_ft': to_tensor(gobs['text_feats'][mask_idx]),
#         }
        
#         if class_name in BG_CLASSES:
#             bg_detection_list.append(detected_object)
#         else:
#             fg_detection_list.append(detected_object)
    
#     return fg_detection_list, bg_detection_list

def transform_detection_list(
    detection_list: DetectionList,
    transform: torch.Tensor,
    deepcopy = False,
):
    '''
    Transform the detection list by the given transform
    
    Args:
        detection_list: DetectionList
        transform: 4x4 torch.Tensor
        
    Returns:
        transformed_detection_list: DetectionList
    '''
    transform = to_numpy(transform)

    if deepcopy:
        detection_list = copy.deepcopy(detection_list)

    for i in range(len(detection_list)):
        detection_list[i]['pcd'] = detection_list[i]['pcd'].transform(transform)
        detection_list[i]['bbox'] = detection_list[i]['bbox'].rotate(transform[:3, :3], center=(0, 0, 0))
        detection_list[i]['bbox'] = detection_list[i]['bbox'].translate(transform[:3, 3])
        # detection_list[i]['bbox'] = detection_list[i]['pcd'].get_oriented_bounding_box(robust=True)

    return detection_list

# @profile
def make_detection_list_from_pcd_and_gobs(
    obj_pcds_and_bboxes, gobs, color_path, obj_classes, image_idx
):
    '''
    This function makes a detection list for the objects
    Ideally I don't want it to be needed, the detection list has too much info and is inefficient
    '''
    global tracker
    detection_list = DetectionList()
    # bg_detection_list = DetectionList()
    for mask_idx in range(len(gobs['mask'])):
        if obj_pcds_and_bboxes[mask_idx] is None: # pointcloud was discarded
            continue

        curr_class_name = gobs['classes'][gobs['class_id'][mask_idx]]
        curr_class_idx = obj_classes.get_classes_arr().index(curr_class_name)
        
        is_bg_object = bool(curr_class_name in obj_classes.get_bg_classes_arr())
        
        tracker.curr_class_count[curr_class_name] += 1
        tracker.total_object_count += 1
        # print(f"Line 937, tracker.total_object_count INCREMENTED: {tracker.total_object_count }")
        num_obj_in_class = tracker.curr_class_count[curr_class_name]
        
        tracker.brand_new_counter += 1
        
        detected_object = {
            'id' : uuid.uuid4(),
            'image_idx' : [image_idx],                             # idx of the image
            
            'mask_idx' : [mask_idx],                         # idx of the mask/detection
            'color_path' : [color_path],                     # path to the RGB image
            'class_name' : curr_class_name,                         # global class id for this detection
            'class_id' : [curr_class_idx],                         # global class id for this detection
            'num_detections' : 1,                            # number of detections in this object
            'mask': gobs['mask'][mask_idx],
            'xyxy': [gobs['xyxy'][mask_idx]],
            'conf': [gobs['confidence'][mask_idx]],
            'n_points': len(obj_pcds_and_bboxes[mask_idx]['pcd'].points),
            # 'pixel_area': [mask.sum()],
            'contain_number': [None],                          # This will be computed later
            "inst_color": np.random.rand(3),                 # A random color used for this segment instance
            'is_background': is_bg_object,
            
            # These are for the entire 3D object
            'pcd': obj_pcds_and_bboxes[mask_idx]['pcd'],
            'bbox': obj_pcds_and_bboxes[mask_idx]['bbox'],
            'clip_ft': to_tensor(gobs['image_feats'][mask_idx]),
            'text_ft': to_tensor(gobs['text_feats'][mask_idx]),
            'num_obj_in_class': num_obj_in_class,
            'curr_obj_num': tracker.total_object_count,
            'new_counter' : tracker.brand_new_counter,
        }
        # detected_object['curr_obj_num']
        # print(f"Line 969, detected_object['image_idx']: {detected_object['image_idx']}")
        # print(f"Line 971, detected_object['class_name']: {detected_object['class_name']}")
        # print(f"Line 966, detected_object['curr_obj_num']: {detected_object['curr_obj_num']}")
        
        # if is_bg_object:
        #     bg_detection_list.append(detected_object)
        # else:
        detection_list.append(detected_object)
    
    return detection_list # , bg_detection_list

# @profile
def dynamic_downsample(points, colors=None, target=5000):
    """
    Simplified and configurable downsampling function that dynamically adjusts the 
    downsampling rate based on the number of input points. If a target of -1 is provided, 
    downsampling is bypassed, returning the original points and colors.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) for N points.
        target (int): Target number of points to aim for in the downsampled output, 
                      or -1 to bypass downsampling.
        colors (torch.Tensor, optional): Corresponding colors tensor of shape (N, 3). 
                                         Defaults to None.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: Downsampled points and optionally 
                                                     downsampled colors, or the original 
                                                     points and colors if target is -1.
    """
    # Check if downsampling is bypassed
    if target == -1:
        return points, colors
    
    num_points = points.size(0)
    
    # If the number of points is less than or equal to the target, return the original points and colors
    if num_points <= target:
        return points, colors
    
    # Calculate downsampling factor to aim for the target number of points
    downsample_factor = max(1, num_points // target)
    
    # Select points based on the calculated downsampling factor
    downsampled_points = points[::downsample_factor]
    
    # If colors are provided, downsample them with the same factor
    downsampled_colors = colors[::downsample_factor] if colors is not None else None

    return downsampled_points, downsampled_colors


def batch_mask_depth_to_points_colors(
    depth_tensor: torch.Tensor,
    masks_tensor: torch.Tensor,
    cam_K: torch.Tensor,
    image_rgb_tensor: torch.Tensor = None,  # Parameter for RGB image tensor
    device: str = 'cuda'
) -> tuple:
    """
    Converts a batch of masked depth images to 3D points and corresponding colors.

    Args:
        depth_tensor (torch.Tensor): A tensor of shape (N, H, W) representing the depth images.
        masks_tensor (torch.Tensor): A tensor of shape (N, H, W) representing the masks for each depth image.
        cam_K (torch.Tensor): A tensor of shape (3, 3) representing the camera intrinsic matrix.
        image_rgb_tensor (torch.Tensor, optional): A tensor of shape (N, H, W, 3) representing the RGB images. Defaults to None.
        device (str, optional): The device to perform the computation on. Defaults to 'cuda'.

    Returns:
        tuple: A tuple containing the 3D points tensor of shape (N, H, W, 3) and the colors tensor of shape (N, H, W, 3).
    """
    N, H, W = masks_tensor.shape
    fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]
    
    # Generate grid of pixel coordinates
    y, x = torch.meshgrid(torch.arange(0, H, device=device), torch.arange(0, W, device=device), indexing='ij')
    z = depth_tensor.repeat(N, 1, 1) * masks_tensor  # Apply masks to depth

    valid = (z > 0).float()  # Mask out zeros

    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    
    points = torch.stack((x, y, z), dim=-1) * valid.unsqueeze(-1)  # Shape: (N, H, W, 3)

    if image_rgb_tensor is not None:
        # Repeat RGB image for each mask and apply masks
        repeated_rgb = image_rgb_tensor.repeat(N, 1, 1, 1) * masks_tensor.unsqueeze(-1)
        colors = repeated_rgb * valid.unsqueeze(-1)  # Apply valid mask to filter out background
    else:
        print("No RGB image provided, assigning random colors to objects")
        # log it as well
        logging.warning("No RGB image provided, assigning random colors to objects")
        # Generate a random color for each mask
        random_colors = torch.randint(0, 256, (N, 3), device=device, dtype=torch.float32) / 255.0  # RGB colors in [0, 1]
        # Expand dims to match (N, H, W, 3) and apply to valid points
        colors = random_colors.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1) * valid.unsqueeze(-1)

    return points, colors


def detections_to_obj_pcd_and_bbox(
    depth_array, 
    masks, 
    cam_K, 
    image_rgb=None, 
    trans_pose=None, 
    min_points_threshold=5, 
    spatial_sim_type='axis_aligned', 
    obj_pcd_max_points = None,
    downsample_voxel_size = None,
    dbscan_remove_noise = None,
    dbscan_eps = None,
    dbscan_min_points = None,
    run_dbscan = None,
    device='cuda'
):
    """
    This function processes a batch of objects to create colored point clouds, apply transformations, and compute bounding boxes.

    Args:
        depth_array (numpy.ndarray): Array containing depth values.
        masks (numpy.ndarray): Array containing binary masks for each object.
        cam_K (numpy.ndarray): Camera intrinsic matrix.
        image_rgb (numpy.ndarray, optional): RGB image. Defaults to None.
        trans_pose (numpy.ndarray, optional): Transformation matrix. Defaults to None.
        min_points_threshold (int, optional): Minimum number of points required for an object. Defaults to 5.
        spatial_sim_type (str, optional): Type of spatial similarity. Defaults to 'axis_aligned'.
        device (str, optional): Device to use. Defaults to 'cuda'.

    Returns:
        list: List of dictionaries containing processed objects. Each dictionary contains a point cloud and a bounding box.
    """
    N, H, W = masks.shape

    # Convert inputs to tensors and move to the specified device
    depth_tensor = torch.from_numpy(depth_array).to(device).float()
    masks_tensor = torch.from_numpy(masks).to(device).float()
    cam_K_tensor = torch.from_numpy(cam_K).to(device).float()

    if image_rgb is not None:
        image_rgb_tensor = torch.from_numpy(image_rgb).to(device).float() / 255.0  # Normalize RGB values
    else:
        image_rgb_tensor = None

    points_tensor, colors_tensor = batch_mask_depth_to_points_colors(
        depth_tensor, masks_tensor, cam_K_tensor, image_rgb_tensor, device
    )

    processed_objects = [None] * N  # Initialize with placeholders
    for i in range(N):
        mask_points = points_tensor[i]
        mask_colors = colors_tensor[i] if colors_tensor is not None else None

        valid_points_mask = mask_points[:, :, 2] > 0
        if torch.sum(valid_points_mask) < min_points_threshold:
            continue

        valid_points = mask_points[valid_points_mask]
        valid_colors = mask_colors[valid_points_mask] if mask_colors is not None else None

        downsampled_points, downsampled_colors = dynamic_downsample(valid_points, colors=valid_colors, target=obj_pcd_max_points)

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(downsampled_points.cpu().numpy())
        if downsampled_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(downsampled_colors.cpu().numpy())

        if trans_pose is not None:
            pcd.transform(trans_pose)  # Apply transformation directly to the point cloud

        bbox = get_bounding_box(spatial_sim_type, pcd)
        if bbox.volume() < 1e-6:
            continue

        processed_objects[i] = {'pcd': pcd, 'bbox': bbox}

    return processed_objects


def processing_needed(
    process_interval, run_on_final_frame, frame_idx, is_final_frame=False
):

    if process_interval > 0 and (frame_idx+1) % process_interval == 0:
        return True
    if run_on_final_frame and is_final_frame:
        return True
    return False


def prepare_objects_save_vis(objects: MapObjectList, downsample_size: float=0.025):
    objects_to_save = copy.deepcopy(objects)
            
    # Downsample the point cloud
    for i in range(len(objects_to_save)):
        objects_to_save[i]['pcd'] = objects_to_save[i]['pcd'].voxel_down_sample(downsample_size)

    # Remove unnecessary keys
    for i in range(len(objects_to_save)):
        for k in list(objects_to_save[i].keys()):
            if k not in [
                'pcd', 'bbox', 'clip_ft', 'text_ft', 'class_id', 'num_detections', 'inst_color'
            ]:
                del objects_to_save[i][k]
                
    return objects_to_save.to_serializable()


def process_cfg(cfg: DictConfig):
    cfg.dataset_root = Path(cfg.dataset_root)
    cfg.dataset_config = Path(cfg.dataset_config)
    
    if cfg.dataset_config.name != "multiscan.yaml":
        # For datasets whose depth and RGB have the same resolution
        # Set the desired image heights and width from the dataset config
        dataset_cfg = omegaconf.OmegaConf.load(cfg.dataset_config)
        if cfg.image_height is None:
            cfg.image_height = dataset_cfg.camera_params.image_height
        if cfg.image_width is None:
            cfg.image_width = dataset_cfg.camera_params.image_width
        print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
    else:
        # For dataset whose depth and RGB have different resolutions
        assert cfg.image_height is not None and cfg.image_width is not None, \
            "For multiscan dataset, image height and width must be specified"

    return cfg

def prepare_objects_save_vis(objects: MapObjectList, downsample_size: float=0.025):
    objects_to_save = copy.deepcopy(objects)
            
    # Downsample the point cloud
    for i in range(len(objects_to_save)):
        objects_to_save[i]['pcd'] = objects_to_save[i]['pcd'].voxel_down_sample(downsample_size)

    # Remove unnecessary keys
    for i in range(len(objects_to_save)):
        for k in list(objects_to_save[i].keys()):
            if k not in [
                'pcd', 'bbox', 'clip_ft', 'text_ft', 'class_id', 'num_detections', 'inst_color'
            ]:
                del objects_to_save[i][k]
                
    return objects_to_save.to_serializable()