from collections import Counter
import copy
import json
from pathlib import Path
import cv2

import numpy as np
from omegaconf import DictConfig
import open3d as o3d
import torch

import torch.nn.functional as F

import faiss

from conceptgraph.utils.general_utils import to_tensor, to_numpy, Timer
from conceptgraph.slam.slam_classes import MapObjectList, DetectionList

from conceptgraph.utils.ious import compute_3d_iou, compute_3d_iou_accurate_batch, mask_subtract_contained, compute_iou_batch
from conceptgraph.dataset.datasets_common import from_intrinsics_matrix

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

def create_object_pcd(depth_array, mask, cam_K, image, obj_color=None) -> o3d.geometry.PointCloud:
    fx, fy, cx, cy = from_intrinsics_matrix(cam_K)
    
    # Also remove points with invalid depth values
    mask = np.logical_and(mask, depth_array > 0)

    if mask.sum() == 0:
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

def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    ### Remove noise via clustering
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

def process_pcd(pcd, downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, run_dbscan=True):
    pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
    
    if dbscan_remove_noise and run_dbscan:
        pcd = pcd_denoise_dbscan(
            pcd, 
            eps=dbscan_eps, 
            min_points=dbscan_min_points
        )
        
    return pcd

def get_bounding_box(spatial_sim_type, pcd):
    if ("accurate" in spatial_sim_type or "overlap" in spatial_sim_type) and len(pcd.points) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()
    else:
        return pcd.get_axis_aligned_bounding_box()

# def merge_obj2_into_obj1(obj1, obj2, spatial_sim_type, device, run_dbscan=True):
def merge_obj2_into_obj1(obj1, obj2, downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, spatial_sim_type, device, run_dbscan=True):

    '''
    Merge the new object to the old object
    This operation is done in-place
    '''
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    
    for k in obj1.keys():
        if k in ['caption']:
            # Here we need to merge two dictionaries and adjust the key of the second one
            for k2, v2 in obj2['caption'].items():
                obj1['caption'][k2 + n_obj1_det] = v2
        elif k not in ['pcd', 'bbox', 'clip_ft', "text_ft"]:
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
    '''
    compute pairwise overlapping between two set of objects in terms of point nearest neighbor. 
    objects_map is the existing objects in the map, objects_new is the new objects to be added to the map
    Suppose len(objects_map) = m, len(objects_new) = n
    Then we want to construct a matrix of size m x n, where the (i, j) entry is the ratio of points 
    in point cloud i that are within a distance threshold of any point in point cloud j.
    '''
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

def merge_overlap_objects(merge_overlap_thresh: float, merge_visual_sim_thresh: float, merge_text_sim_thresh: float, objects: MapObjectList, overlap_matrix: np.ndarray, downsample_voxel_size: float, dbscan_remove_noise: bool, dbscan_eps: float, dbscan_min_points: int, spatial_sim_type: str, device: str):
    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]

    sort = np.argsort(overlap_ratio)[::-1]  # Sort indices of overlap ratios in descending order
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]

    kept_objects = np.ones(len(objects), dtype=bool)  # Initialize all objects as 'kept' initially
    for i, j, ratio in zip(x, y, overlap_ratio):
        if ratio > merge_overlap_thresh:
            visual_sim = F.cosine_similarity(
                to_tensor(objects[i]['clip_ft']),
                to_tensor(objects[j]['clip_ft']),
                dim=0
            )
            text_sim = F.cosine_similarity(
                to_tensor(objects[i]['text_ft']),
                to_tensor(objects[j]['text_ft']),
                dim=0
            )
            if visual_sim > merge_visual_sim_thresh and text_sim > merge_text_sim_thresh:
                if kept_objects[j]:  # Check if the target object has not been merged into another
                    # Merge object i into object j
                    objects[j] = merge_obj2_into_obj1(
                        objects[j], objects[i], 
                        downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, 
                        spatial_sim_type, device, 
                        run_dbscan=True
                    )
                    kept_objects[i] = False  # Mark object i as 'merged'
        else:
            break  # Stop processing if the current overlap ratio is below the threshold

    # Create a new list of objects excluding those that were merged
    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)
    
    return objects

def denoise_objects(downsample_voxel_size: float, dbscan_remove_noise: bool, dbscan_eps: float, dbscan_min_points: int, spatial_sim_type: str, device: str, objects: MapObjectList):
    for i in range(len(objects)):
        og_object_pcd = objects[i]['pcd']
        # Adjust the call to process_pcd with explicit parameters
        objects[i]['pcd'] = process_pcd(objects[i]['pcd'], downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, run_dbscan=True)
        if len(objects[i]['pcd'].points) < 4:
            objects[i]['pcd'] = og_object_pcd
            continue
        # Adjust the call to get_bounding_box with explicit parameters
        objects[i]['bbox'] = get_bounding_box(spatial_sim_type, objects[i]['pcd'])
        objects[i]['bbox'].color = [0,1,0]
        
    return objects


def filter_objects(obj_min_points: int, obj_min_detections: int, objects: MapObjectList):
    print("Before filtering:", len(objects))
    objects_to_keep = []
    for obj in objects:
        if len(obj['pcd'].points) >= obj_min_points and obj['num_detections'] >= obj_min_detections:
            objects_to_keep.append(obj)
    objects = MapObjectList(objects_to_keep)
    print("After filtering:", len(objects))
    
    return objects
    

def merge_objects(merge_overlap_thresh: float, merge_visual_sim_thresh: float, merge_text_sim_thresh: float, objects: MapObjectList, downsample_voxel_size: float, dbscan_remove_noise: bool, dbscan_eps: float, dbscan_min_points: int, spatial_sim_type: str, device: str):
    if merge_overlap_thresh > 0:
        # Assuming compute_overlap_matrix requires only `objects` and `downsample_voxel_size`
        overlap_matrix = compute_overlap_matrix(objects, downsample_voxel_size)
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
            device=device
        )
        print("After merging:", len(objects))

    return objects

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
        if gobs['mask'][mask_idx].sum() < max(mask_area_threshold, 10):
            continue
        
        # Skip the BG classes
        if skip_bg and class_name in BG_CLASSES:
            continue
        
        # Skip the non-background boxes that are too large
        if class_name not in BG_CLASSES:
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            if max_bbox_area_ratio is not None and bbox_area > max_bbox_area_ratio * image_area:
                continue
            
        # Skip masks with low confidence
        if mask_conf_threshold is not None and gobs['confidence'] is not None:
            if gobs['confidence'][mask_idx] < mask_conf_threshold:
                continue
        
        idx_to_keep.append(mask_idx)
    
    for k in gobs.keys():
        if isinstance(gobs[k], str) or k == "classes": # Captions
            continue
        elif isinstance(gobs[k], list):
            gobs[k] = [gobs[k][i] for i in idx_to_keep]
        elif isinstance(gobs[k], np.ndarray):
            gobs[k] = gobs[k][idx_to_keep]
        else:
            raise NotImplementedError(f"Unhandled type {type(gobs[k])}")
    
    return gobs

def resize_gobs(
    gobs,
    image
):
    n_masks = len(gobs['xyxy'])

    new_mask = []
    
    for mask_idx in range(n_masks):
        # TODO: rewrite using interpolation/resize in numpy or torch rather than cv2
        mask = gobs['mask'][mask_idx]
        if mask.shape != image.shape[:2]:
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
            new_mask.append(mask)

    if len(new_mask) > 0:
        gobs['mask'] = np.asarray(new_mask)
        
    return gobs

def gobs_to_detection_list(
    image, 
    depth_array,
    cam_K, 
    idx, 
    gobs, 
    trans_pose = None,
    class_names = None,
    BG_CLASSES  = None,
    skip_bg = None,
    color_path = None,
    mask_area_threshold: int = None,
    max_bbox_area_ratio: float = None,
    mask_conf_threshold: float = None,
    min_points_threshold: int = None,
    spatial_sim_type: str = None,
    downsample_voxel_size: float = None,  # New parameter
    dbscan_remove_noise: bool = None,     # New parameter
    dbscan_eps: float = None,             # New parameter
    dbscan_min_points: int = None         # New parameter
):
    '''
    Return a DetectionList object from the gobs
    All object are still in the camera frame. 
    '''
    fg_detection_list = DetectionList()
    bg_detection_list = DetectionList()
    
    gobs = resize_gobs(gobs, image)
    gobs = filter_gobs(
        gobs, 
        image, 
        skip_bg=skip_bg,
        BG_CLASSES=BG_CLASSES,
        mask_area_threshold=mask_area_threshold,
        max_bbox_area_ratio=max_bbox_area_ratio,
        mask_conf_threshold=mask_conf_threshold,
    )
    
    if len(gobs['xyxy']) == 0:
        return fg_detection_list, bg_detection_list
    
    # Compute the containing relationship among all detections and subtract fg from bg objects
    xyxy = gobs['xyxy']
    mask = gobs['mask']
    gobs['mask'] = mask_subtract_contained(xyxy, mask)
    
    n_masks = len(gobs['xyxy'])
    for mask_idx in range(n_masks):
        local_class_id = gobs['class_id'][mask_idx]
        mask = gobs['mask'][mask_idx]
        class_name = gobs['classes'][local_class_id]
        global_class_id = -1 if class_names is None else class_names.index(class_name)
        
        # make the pcd and color it
        camera_object_pcd = create_object_pcd(
            depth_array,
            mask,
            cam_K,
            image,
            obj_color = None
        )
        
        # It at least contains 5 points
        if len(camera_object_pcd.points) < max(min_points_threshold, 5): 
            continue
        
        if trans_pose is not None:
            global_object_pcd = camera_object_pcd.transform(trans_pose)
        else:
            global_object_pcd = camera_object_pcd
        
        # get largest cluster, filter out noise 
        # global_object_pcd = process_pcd(global_object_pcd, cfg)
        global_object_pcd = process_pcd(global_object_pcd, downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, run_dbscan=True)

        
        # pcd_bbox = get_bounding_box(cfg, global_object_pcd)
        pcd_bbox = get_bounding_box(spatial_sim_type, global_object_pcd)
        pcd_bbox.color = [0,1,0]
        
        if pcd_bbox.volume() < 1e-6:
            continue
        
        # Treat the detection in the same way as a 3D object
        # Store information that is enough to recover the detection
        detected_object = {
            'image_idx' : [idx],                             # idx of the image
            'mask_idx' : [mask_idx],                         # idx of the mask/detection
            'color_path' : [color_path],                     # path to the RGB image
            'class_name' : [class_name],                         # global class id for this detection
            'class_id' : [global_class_id],                         # global class id for this detection
            'num_detections' : 1,                            # number of detections in this object
            'mask': [mask],
            'xyxy': [gobs['xyxy'][mask_idx]],
            'conf': [gobs['confidence'][mask_idx]],
            'n_points': [len(global_object_pcd.points)],
            'pixel_area': [mask.sum()],
            'contain_number': [None],                          # This will be computed later
            "inst_color": np.random.rand(3),                 # A random color used for this segment instance
            'is_background': class_name in BG_CLASSES,
            
            # These are for the entire 3D object
            'pcd': global_object_pcd,
            'bbox': pcd_bbox,
            'clip_ft': to_tensor(gobs['image_feats'][mask_idx]),
            'text_ft': to_tensor(gobs['text_feats'][mask_idx]),
        }
        
        if class_name in BG_CLASSES:
            bg_detection_list.append(detected_object)
        else:
            fg_detection_list.append(detected_object)
    
    return fg_detection_list, bg_detection_list

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