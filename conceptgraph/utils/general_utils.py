from copy import deepcopy
import gzip
import json
import logging
import os
from pathlib import Path
import pickle
# from conceptgraph.utils.vis import annotate_for_vlm, filter_detections, plot_edges_from_vlm
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.utils import prepare_objects_save_vis
from conceptgraph.utils.ious import mask_subtract_contained
import supervision as sv
import scipy.ndimage as ndi 
from conceptgraph.utils.vlm import get_obj_rel_from_image_gpt4v
import cv2


from omegaconf import OmegaConf
import torch
import numpy as np
import time

class Timer:
    def __init__(self, heading = "", verbose = True):
        self.verbose = verbose
        if not self.verbose:
            return
        self.heading = heading

    def __enter__(self):
        if not self.verbose:
            return self
        self.start = time.time()
        return self

    def __exit__(self, *args):
        if not self.verbose:
            return
        self.end = time.time()
        self.interval = self.end - self.start
        print(self.heading, self.interval)




def prjson(input_json, indent=0):
    """ Pretty print a json object """
    if not isinstance(input_json, list):
        input_json = [input_json]
        
    print("[")
    for i, entry in enumerate(input_json):
        print("  {")
        for j, (key, value) in enumerate(entry.items()):
            terminator = "," if j < len(entry) - 1 else ""
            if isinstance(value, str):
                formatted_value = value.replace("\\n", "\n").replace("\\t", "\t")
                print('    "{}": "{}"{}'.format(key, formatted_value, terminator))
            else:
                print(f'    "{key}": {value}{terminator}')
        print("  }" + ("," if i < len(input_json) - 1 else ""))
    print("]")

def cfg_to_dict(input_cfg):
    """ Convert a Hydra configuration object to a native Python dictionary,
    ensuring all special types (e.g., ListConfig, DictConfig, PosixPath) are
    converted to serializable types for JSON. Checks for non-serializable objects. """
    
    def convert_to_serializable(obj):
        """ Recursively convert non-serializable objects to serializable types. """
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    def check_serializability(obj, context=""):
        """ Attempt to serialize the object, raising an error if not possible. """
        try:
            json.dumps(obj)
        except TypeError as e:
            raise TypeError(f"Non-serializable object encountered in {context}: {e}")

        if isinstance(obj, dict):
            for k, v in obj.items():
                check_serializability(v, context=f"{context}.{k}" if context else str(k))
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                check_serializability(item, context=f"{context}[{idx}]")

    # Convert Hydra configs to native Python types
    # check if its already a dictionary, in which case we don't need to convert it
    if not isinstance(input_cfg, dict):
        native_cfg = OmegaConf.to_container(input_cfg, resolve=True)
    else:
        native_cfg = input_cfg
    # Convert all elements to serializable types
    serializable_cfg = convert_to_serializable(native_cfg)
    # Check for serializability of the entire config
    check_serializability(serializable_cfg)

    return serializable_cfg

def get_stream_data_out_path(dataset_root, scene_id, make_dir=True):
    stream_data_out_path = Path(dataset_root) / scene_id
    stream_rgb_path = stream_data_out_path / "rgb"
    stream_depth_path = stream_data_out_path / "depth"
    stream_poses_path = stream_data_out_path / "poses"
    
    if make_dir:
        stream_rgb_path.mkdir(parents=True, exist_ok=True)
        stream_depth_path.mkdir(parents=True, exist_ok=True)
        stream_poses_path.mkdir(parents=True, exist_ok=True)
        
    return stream_rgb_path, stream_depth_path, stream_poses_path

def get_exp_out_path(dataset_root, scene_id, exp_suffix, make_dir=True):
    exp_out_path = Path(dataset_root) / scene_id / "exps" / f"{exp_suffix}"
    if make_dir:
        exp_out_path.mkdir(exist_ok=True, parents=True)
    return exp_out_path

def get_vis_out_path(exp_out_path):
    vis_folder_path = exp_out_path / "vis"
    vis_folder_path.mkdir(exist_ok=True, parents=True)
    return vis_folder_path

def get_det_out_path(exp_out_path, make_dir=True):
    detections_folder_path = exp_out_path / "detections"
    if make_dir:
        detections_folder_path.mkdir(exist_ok=True, parents=True)
    return detections_folder_path

def check_run_detections(force_detection, det_exp_path):
    # first check if det_exp_path directory exists
    if force_detection:
        return True
    if not det_exp_path.exists():
        return True
    return False

def mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

def annotate_for_vlm(
    image: np.ndarray, 
    detections: sv.Detections,
    obj_classes, 
    labels: list[str], 
    save_path=None, 
    color: tuple=(0, 255, 0), 
    thickness: int=2, 
    text_color: tuple=(255, 255, 255), 
    text_scale: float=0.5, 
    text_thickness: int=3, 
    text_bg_color: tuple=(255, 255, 255), 
    text_bg_opacity: float=0.95,  # Opacity from 0 (transparent) to 1 (opaque)
    small_mask_threshold = 0.002,
    mask_opacity: float = 0.2  # Opacity for mask fill
) -> np.ndarray:
    annotated_image = image.copy()
    
    
    # if image.shape[0] > 700:
    #     print(f"Line 604, image.shape[0]: {image.shape[0]}")
    #     text_scale = 2.5
    #     text_thickness = 5
    total_pixels = image.shape[0] * image.shape[1]
    small_mask_size = total_pixels * small_mask_threshold
    
    detections_mask = detections.mask
    detections_mask = mask_subtract_contained(detections.xyxy, detections_mask)
    
    # Sort detections by mask area, large to small, and keep track of original indices
    mask_areas = [np.count_nonzero(mask) for mask in detections_mask]
    sorted_indices = sorted(range(len(mask_areas)), key=lambda x: mask_areas[x], reverse=True)
    
    # Iterate over each mask and corresponding label in the detections in sorted order
    for i in sorted_indices:
        mask = detections_mask[i]
        label = labels[i]
        label_num = label.split(" ")[-1]
        bbox = detections.xyxy[i]
        
        obj_color = obj_classes.get_class_color(int(detections.class_id[i]))
        # multiply by 255 to convert to BGR
        obj_color = tuple([int(c * 255) for c in obj_color])
        
        # Convert mask to uint8 type
        mask_uint8 = mask.astype(np.uint8)
        mask_color_image = np.zeros_like(annotated_image)
        mask_color_image[mask_uint8 > 0] = obj_color
        cv2.addWeighted(annotated_image, 1, mask_color_image, mask_opacity, 0, annotated_image)

        # Draw contours
        contours, _ = cv2.findContours(mask_uint8 * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated_image, contours, -1, obj_color, thickness)

        # Determine if the mask is considered "small"
        if mask_areas[i] < small_mask_size:
            x_center = int(bbox[2])  # Place the text to the right of the bounding box
            y_center = int(bbox[1])  # Place the text above the top of the bounding box
        else:
            # Calculate the centroid of the mask
            ys, xs = np.nonzero(mask)
            y_center, x_center = ndi.center_of_mass(mask)
            x_center, y_center = int(x_center), int(y_center)

        # Prepare text background
        text = label_num
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
        text_x_left = x_center - text_width // 2
        text_y_top = y_center + (text_height) // 2
        
        # Create a rectangle sub-image for the text background
        b_pad = 2 # background rectangle padding
        rect_top_left = (text_x_left - b_pad, text_y_top - text_height - baseline - b_pad)
        rect_bottom_right = (text_x_left + text_width + b_pad, text_y_top - baseline//2 + b_pad)
        sub_img = annotated_image[rect_top_left[1]:rect_bottom_right[1], rect_top_left[0]:rect_bottom_right[0]]
        
        # Create the background rectangle with the specified color and opacity
        # make the text bg color be the negative of the text color
        text_bg_color = tuple([255 - c for c in obj_color])
        # now make text bg color grayscale
        text_bg_color = tuple([int(sum(text_bg_color) / 3)] * 3)
        background_rect = np.full(sub_img.shape, text_bg_color, dtype=np.uint8)
        # cv2.addWeighted(sub_img, 1 - text_bg_opacity, background_rect, text_bg_opacity, 0, sub_img)

        # Draw text with background
        cv2.putText(
            annotated_image, 
            text, 
            (text_x_left, text_y_top - baseline), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            text_scale, 
            # obj_color,
            # (255,255,255),
            (0,0,0),
            text_thickness, 
            cv2.LINE_AA
        )
        
        # Draw text with background
        cv2.putText(
            annotated_image, 
            text, 
            (text_x_left, text_y_top - baseline), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            text_scale,
            # (0,0,0), 
            obj_color,
            text_thickness - 1, 
            cv2.LINE_AA
        )
        
        if save_path:
            cv2.imwrite(save_path, annotated_image)

    return annotated_image, sorted_indices

def plot_edges_from_vlm(image: np.ndarray, edges, detections: sv.Detections, obj_classes, labels: list[str], sorted_indices: list[int], save_path=None) -> np.ndarray:
    annotated_image = image.copy()
    
    # Create a map from label to mask centroid and color for quick lookup
    label_to_centroid_color = {}
    for idx in sorted_indices:
        mask = detections.mask[idx]
        label_num = labels[idx].split(' ')[-1]  # Assuming label format is 'object X'
        obj_color = obj_classes.get_class_color(int(detections.class_id[idx]))
        obj_color = tuple([int(c * 255) for c in obj_color])  # Convert to BGR
    
        # Determine the centroid of the mask
        ys, xs = np.nonzero(mask)
        if ys.size > 0 and xs.size > 0:
            y_center, x_center = ndi.center_of_mass(mask)
            centroid = (int(x_center), int(y_center))
        else:
            # Fallback to bbox center if mask is empty
            bbox = detections.xyxy[idx]
            centroid = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        
        label_to_centroid_color[label_num] = (centroid, obj_color)
    
    # Draw edges based on relationships specified
    for edge in edges:
        src_label, _, dst_label = edge
        src_label = src_label.split(' ')[-1]  # Assuming label format is 'object X'
        dst_label = dst_label.split(' ')[-1]  # Assuming label format is 'object X'
        if src_label in label_to_centroid_color and dst_label in label_to_centroid_color:
            src_centroid, _ = label_to_centroid_color[src_label]
            dst_centroid, dst_color = label_to_centroid_color[dst_label]
            # Draw line from source to destination object with the color of the destination object
            cv2.line(annotated_image, src_centroid, dst_centroid, dst_color, 2)
    
    if save_path:
            cv2.imwrite(str(save_path), annotated_image)
            
    return annotated_image

def filter_detections(
    image,
    detections: sv.Detections, 
    classes, 
    top_x_detections = None, 
    confidence_threshold: float = 0.0,
    given_labels = None,
    iou_threshold: float = 0.80,  # IoU similarity threshold
    proximity_threshold: float = 20.0,  # Default proximity threshold
    keep_larger: bool = True,  # Keep the larger bounding box by area if True, else keep the smaller
    min_mask_size_ratio=0.00025
) -> tuple[sv.Detections, list[str]]:
    '''
    Filter detections based on confidence, top X detections, and proximity of bounding boxes.
    Args:
        proximity_threshold (float): The minimum distance between centers of bounding boxes to consider them non-overlapping.
        keep_larger (bool): If True, keeps the larger bounding box when overlaps occur; otherwise keeps the smaller.
    Returns:
        tuple[sv.Detections, list[str]]: Filtered detections and labels.
    '''
    if not (hasattr(detections, 'confidence') and hasattr(detections, 'class_id') and hasattr(detections, 'xyxy')):
        print("Detections object is missing required attributes.")
        return detections, []

    # Sort by confidence initially
    detections_combined = sorted(
        zip(detections.confidence, detections.class_id, detections.xyxy, detections.mask, range(len(given_labels))),
        key=lambda x: x[0], reverse=True
    )

    if top_x_detections is not None:
        detections_combined = detections_combined[:top_x_detections]

    # Further filter based on proximity
    filtered_detections = []
    for idx, current_det in enumerate(detections_combined):
        _, curr_class_id, curr_xyxy, curr_mask, _ = current_det
        curr_center = ((curr_xyxy[0] + curr_xyxy[2]) / 2, (curr_xyxy[1] + curr_xyxy[3]) / 2)
        curr_area = (curr_xyxy[2] - curr_xyxy[0]) * (curr_xyxy[3] - curr_xyxy[1])
        keep = True
        
            # Calculate the total number of pixels as a threshold for small masks
        total_pixels = image.shape[0] * image.shape[1]
        small_mask_size = total_pixels * min_mask_size_ratio

        # check mask size and remove if too small
        mask_size = np.count_nonzero(current_det[3])
        if mask_size < small_mask_size:
            print(f"Removing {classes.get_classes_arr()[curr_class_id]} because the mask size is too small.")
            keep = False

        for other in filtered_detections:
            _, other_class_id, other_xyxy, other_mask, _ = other
            
            if mask_iou(curr_mask, other_mask) > iou_threshold:
                keep = False
                break
            
            
            other_center = ((other_xyxy[0] + other_xyxy[2]) / 2, (other_xyxy[1] + other_xyxy[3]) / 2)
            other_area = (other_xyxy[2] - other_xyxy[0]) * (other_xyxy[3] - other_xyxy[1])

            # Calculate distance between centers
            dist = np.sqrt((curr_center[0] - other_center[0]) ** 2 + (curr_center[1] - other_center[1]) ** 2)
            if dist < proximity_threshold:
                if (keep_larger and curr_area > other_area) or (not keep_larger and curr_area < other_area):
                    filtered_detections.remove(other)
                else:
                    keep = False
                    break
        # print(given_labels[idx])
        if classes.get_classes_arr()[curr_class_id] in classes.bg_classes:
            print(f"Removing {classes.get_classes_arr()[curr_class_id]} because it is a background class, specifically {classes.bg_classes}.")
            keep = False

        if keep:
            filtered_detections.append(current_det)

    # Unzip the filtered results
    confidences, class_ids, xyxy, masks, indices = zip(*filtered_detections)
    filtered_labels = [given_labels[i] for i in indices]

    # Create new detections object
    filtered_detections = sv.Detections(
        class_id=np.array(class_ids, dtype=np.int64),
        confidence=np.array(confidences, dtype=np.float32),
        xyxy=np.array(xyxy, dtype=np.float32),
        mask=np.array(masks, dtype=np.bool_)
    )

    return filtered_detections, filtered_labels

def get_vlm_annotated_image_path(det_exp_vis_path, color_path, w_edges=False, suffix="annotated_for_vlm.jpg", ):

    # Define suffixes based on whether edges are included
    if w_edges:
        suffix = suffix.replace(".jpg", "_w_edges.jpg")

    # Create the file path
    vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg").with_name(
        (det_exp_vis_path / color_path.name).stem + suffix
    )
    return str(vis_save_path)

def make_vlm_edges(image, curr_det, obj_classes, detection_class_labels, det_exp_vis_path, color_path, make_edges_flag, openai_client):
    """
    Process detections by filtering, annotating, and extracting object relationships.

    Args:
    image: The image on which detections are performed.
    curr_det: Current detections from the detection model.
    obj_classes: Object classes used in detection.
    detection_class_labels: Labels for each detection class.
    det_exp_vis_path: Directory path for saving visualizations.
    color_path: Additional path element for creating unique save paths.
    cfg: Configuration object containing settings like `make_edges`.
    openai_client: Client object for OpenAI used in relationship extraction.

    Returns:
    detection_class_labels: The original labels provided for detection classes.
    labels: The labels after filtering detections.
    edges: List of edges between detected objects if `make_edges` is true, otherwise empty list.
    """
    # Filter the detections
    filtered_detections, labels = filter_detections(
        image=image,
        detections=curr_det, 
        classes=obj_classes,
        top_x_detections=150000,
        confidence_threshold=0.00001,
        given_labels=detection_class_labels,
    )
    
    edges = []
    edge_image = None
    if make_edges_flag:
        vis_save_path_for_vlm = get_vlm_annotated_image_path(det_exp_vis_path, color_path)
        vis_save_path_for_vlm_edges = get_vlm_annotated_image_path(det_exp_vis_path, color_path, w_edges=True)
        annotated_image_for_vlm, sorted_indices = annotate_for_vlm(image, filtered_detections, obj_classes, labels, save_path=vis_save_path_for_vlm)

        label_nums = [f"object {str(label.split(' ')[-1])}" for label in labels]
        cv2.imwrite(str(vis_save_path_for_vlm), annotated_image_for_vlm)
        print(f"Line 313, vis_save_path_for_vlm: {vis_save_path_for_vlm}")
        
        edges = get_obj_rel_from_image_gpt4v(openai_client, vis_save_path_for_vlm, label_nums)
        edge_image = plot_edges_from_vlm(annotated_image_for_vlm, edges, filtered_detections, obj_classes, labels, sorted_indices, save_path=vis_save_path_for_vlm_edges)
    
    return labels, edges, edge_image
    
def handle_rerun_saving(use_rerun, save_rerun, exp_suffix, exp_out_path):
    # Save the rerun output if needed
    if use_rerun and save_rerun:
        rerun_file_path = exp_out_path / f"rerun_{exp_suffix}.rrd"
        print("Mapping done!")
        print("If you want to save the rerun file, you should do so from the rerun viewer now.")
        print("You can't yet both save and log a file in rerun.")
        print("If you do, make a pull request!")
        print("Also, close the viewer before continuing, it frees up a lot of RAM, which helps for saving the pointclouds.")
        print(f"Feel free to copy and use this path below, or choose your own:\n{rerun_file_path}")
        input("Then press Enter to continue.")

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        # print(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)  # Call the function with any arguments it was called with
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done! Execution time of {func.__name__} function: {elapsed_time:.2f} seconds")
        return result  # Return the result of the function call
    return wrapper

def get_exp_config_save_path(exp_out_path, is_detection_config=False):
    params_file_name = "config_params"
    if is_detection_config:
        params_file_name += "_detections"
    return exp_out_path / f"{params_file_name}.json"

def save_hydra_config(hydra_cfg, exp_out_path, is_detection_config=False):
    exp_out_path.mkdir(exist_ok=True, parents=True)
    with open(get_exp_config_save_path(exp_out_path, is_detection_config), "w") as f:
        dict_to_dump = cfg_to_dict(hydra_cfg)
        json.dump(dict_to_dump, f, indent=2)

def load_saved_hydra_json_config(exp_out_path):
    with open(get_exp_config_save_path(exp_out_path), "r") as f:
        return json.load(f)


def prepare_detection_paths(dataset_root, scene_id, detections_exp_suffix, force_detection, output_base_path):
    """
    Prepare and return paths needed for detection output, creating directories as needed.
    """
    det_exp_path = get_exp_out_path(dataset_root, scene_id, detections_exp_suffix)
    if force_detection:
        det_vis_folder_path = get_vis_out_path(det_exp_path)
        det_detections_folder_path = get_det_out_path(det_exp_path)
        os.makedirs(det_vis_folder_path, exist_ok=True)
        os.makedirs(det_detections_folder_path, exist_ok=True)
        return det_exp_path, det_vis_folder_path, det_detections_folder_path
    return det_exp_path

def should_exit_early(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Check if we should exit early
        if data.get("exit_early", False):
            # Reset the exit_early flag to False
            data["exit_early"] = False
            # Write the updated data back to the file
            with open(file_path, 'w') as file:
                json.dump(data, file)
            return True
        else:
            return False
    except Exception as e:
        # If there's an error reading the file or the key doesn't exist, 
        # log the error and return False
        print(f"Error reading {file_path}: {e}")
        logging.info(f"Error reading {file_path}: {e}")
        return False

def save_detection_results(base_path, results):
    base_path.mkdir(exist_ok=True, parents=True)
    for key, value in results.items():
        save_path = Path(base_path) / f"{key}"
        if isinstance(value, np.ndarray):
            # Save NumPy arrays using .npz for efficient storage
            np.savez_compressed(f"{save_path}.npz", value)
        else:
            # For other types, fall back to pickle
            with gzip.open(f"{save_path}.pkl.gz", "wb") as f:
                pickle.dump(value, f)
                
def load_saved_detections(base_path):
    base_path = Path(base_path)
    
    # Construct potential .pkl.gz file path based on the base_path
    potential_pkl_gz_path = Path(str(base_path) + '.pkl.gz')

    # Check if the constructed .pkl.gz file exists
    # This is the old wat 
    if potential_pkl_gz_path.exists() and potential_pkl_gz_path.is_file():
        # The path points directly to a .pkl.gz file
        with gzip.open(potential_pkl_gz_path, "rb") as f:
            return pickle.load(f)
    elif base_path.is_dir():
        loaded_detections = {}
        for file_path in base_path.iterdir():
            # Handle files based on their extension, adjusting the key extraction method
            if file_path.suffix == '.npz':
                key = file_path.name.replace('.npz', '')
                with np.load(file_path, allow_pickle=True) as data:
                    loaded_detections[key] = data['arr_0']
            elif file_path.suffix == '.gz' and file_path.suffixes[-2] == '.pkl':
                key = file_path.name.replace('.pkl.gz', '')
                with gzip.open(file_path, "rb") as f:
                    loaded_detections[key] = pickle.load(f)
        return loaded_detections
    else:
        raise FileNotFoundError(f"No valid file or directory found at {base_path}")
        
        
class ObjectClasses:
    """
    Manages object classes and their associated colors, allowing for exclusion of background classes.

    This class facilitates the creation or loading of a color map from a specified file containing
    class names. It also manages background classes based on configuration, allowing for their
    inclusion or exclusion. Background classes are ["wall", "floor", "ceiling"] by default.

    Attributes:
        classes_file_path (str): Path to the file containing class names, one per line.

    Usage:
        obj_classes = ObjectClasses(classes_file_path, skip_bg=True)
        model.set_classes(obj_classes.get_classes_arr())
        some_class_color = obj_classes.get_class_color(index or class_name)
    """
    def __init__(self, classes_file_path, bg_classes, skip_bg):
        self.classes_file_path = Path(classes_file_path)
        self.bg_classes = bg_classes
        self.skip_bg = skip_bg
        self.classes, self.class_to_color = self._load_or_create_colors()

    def _load_or_create_colors(self):
        with open(self.classes_file_path, "r") as f:
            all_classes = [cls.strip() for cls in f.readlines()]
        
        # Filter classes based on the skip_bg parameter
        if self.skip_bg:
            classes = [cls for cls in all_classes if cls not in self.bg_classes]
        else:
            classes = all_classes

        colors_file_path = self.classes_file_path.parent / f"{self.classes_file_path.stem}_colors.json"
        if colors_file_path.exists():
            with open(colors_file_path, "r") as f:
                class_to_color = json.load(f)
            # Ensure color map only includes relevant classes
            class_to_color = {cls: class_to_color[cls] for cls in classes if cls in class_to_color}
        else:
            class_to_color = {class_name: list(np.random.rand(3).tolist()) for class_name in classes}
            with open(colors_file_path, "w") as f:
                json.dump(class_to_color, f)

        return classes, class_to_color

    def get_classes_arr(self):
        """
        Returns the list of class names, excluding background classes if configured to do so.
        """
        return self.classes
    
    def get_bg_classes_arr(self):
        """
        Returns the list of background class names, if configured to do so.
        """
        return self.bg_classes

    def get_class_color(self, key):
        """
        Retrieves the color associated with a given class name or index.
        
        Args:
            key (int or str): The index or name of the class.
        
        Returns:
            list: The color (RGB values) associated with the class.
        """
        if isinstance(key, int):
            if key < 0 or key >= len(self.classes):
                raise IndexError("Class index out of range.")
            class_name = self.classes[key]
        elif isinstance(key, str):
            class_name = key
            if class_name not in self.classes:
                raise ValueError(f"{class_name} is not a valid class name.")
        else:
            raise ValueError("Key must be an integer index or a string class name.")
        return self.class_to_color.get(class_name, [0, 0, 0])  # Default color for undefined classes

    def get_class_color_dict_by_index(self):
        """
        Returns a dictionary of class colors, just like self.class_to_color, but indexed by class index.
        """
        return {str(i): self.get_class_color(i) for i in range(len(self.classes))}


def save_pointcloud(exp_suffix, exp_out_path, cfg, objects, obj_classes, latest_pcd_filepath=None, create_symlink=True, edges = None):
    """
    Saves the point cloud data to a .pkl.gz file. Optionally, creates or updates a symlink to the latest saved file.

    Args:
    - exp_suffix (str): Suffix for the experiment, used in naming the saved file.
    - exp_out_path (Path or str): Output path for the experiment's saved files.
    - objects: The objects to save, assumed to have a `to_serializable()` method.
    - obj_classes: The object classes, assumed to have `get_classes_arr()` and `get_class_color_dict_by_index()` methods.
    - latest_pcd_filepath (Path or str, optional): Path for the symlink to the latest point cloud save. Default is None.
    - create_symlink (bool): Whether to create/update a symlink to the latest save. Default is True.
    """
    print("saving map...")
    # Prepare the results dictionary
    results = {
        'objects': objects.to_serializable(),
        'cfg': cfg_to_dict(cfg),
        'class_names': obj_classes.get_classes_arr(),
        'class_colors': obj_classes.get_class_color_dict_by_index(),
        'edges': edges.to_serializable() if edges is not None else None,
    }

    # Define the save path for the point cloud
    pcd_save_path = Path(exp_out_path) / f"pcd_{exp_suffix}.pkl.gz"
    # Make the directory if it doesn't exist
    pcd_save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the point cloud data
    with gzip.open(pcd_save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved point cloud to {pcd_save_path}")
    if edges is not None:
        print(f"Also saved edges to {pcd_save_path}")

    # Create or update the symlink if requested
    if create_symlink and latest_pcd_filepath:
        latest_pcd_path = Path(latest_pcd_filepath)
        # Remove the existing symlink if it exists
        if latest_pcd_path.is_symlink() or latest_pcd_path.exists():
            latest_pcd_path.unlink()
        # Create a new symlink pointing to the latest point cloud save
        latest_pcd_path.symlink_to(pcd_save_path)
        print(f"Updated symlink to point to the latest point cloud save at {latest_pcd_path} to:\n{pcd_save_path}")

        
def find_existing_image_path(base_path, extensions):
    """
    Checks for the existence of a file with the given base path and any of the provided extensions.
    Returns the path of the first existing file found or None if no file is found.

    Parameters:
    - base_path: The base file path without the extension.
    - extensions: A list of file extensions to check for.

    Returns:
    - Path of the existing file or None if no file exists.
    """
    for ext in extensions:
        potential_path = base_path.with_suffix(ext)
        if potential_path.exists():
            return potential_path
    return None

def save_objects_for_frame(obj_all_frames_out_path, frame_idx, objects, obj_min_detections, adjusted_pose, color_path):
    save_path = obj_all_frames_out_path / f"{frame_idx:06d}.pkl.gz"
    filtered_objects = [obj for obj in objects if obj['num_detections'] >= obj_min_detections]
    prepared_objects = prepare_objects_save_vis(MapObjectList(filtered_objects))
    result = {
        "camera_pose": adjusted_pose, 
        "objects": prepared_objects,
        "frame_idx": frame_idx,
        "num_objects": len(filtered_objects),
        "color_path": str(color_path)
    }
    with gzip.open(save_path, 'wb') as f:
        pickle.dump(result, f)
        
def add_info_to_image(image, frame_idx, num_objects, color_path):
    frame_info_text = f"Frame: {frame_idx}, Objects: {num_objects}, Path: {str(color_path)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 0, 0)
    thickness = 1
    line_type = cv2.LINE_AA
    position = (10, image.shape[0] - 10)
    cv2.putText(image, frame_info_text, position, font, font_scale, color, thickness, line_type)
        
def save_video_from_frames(frames, exp_out_path, exp_suffix):
    video_save_path = exp_out_path / (f"s_mapping_{exp_suffix}.mp4")
    save_video_from_frames(frames, video_save_path, fps=10)
    print(f"Save video to {video_save_path}")
        
def vis_render_image(objects, obj_classes, obj_renderer, image_original_pil, adjusted_pose, frames, frame_idx, color_path, obj_min_detections, class_agnostic, debug_render, is_final_frame, exp_out_path, exp_suffix):
    filtered_objects = [
        deepcopy(obj) for obj in objects 
        if obj['num_detections'] >= obj_min_detections and not obj['is_background']
    ]
    objects_vis = MapObjectList(filtered_objects)

    if class_agnostic:
        objects_vis.color_by_instance()
    else:
        objects_vis.color_by_most_common_classes(obj_classes)

    rendered_image, vis = obj_renderer.step(
        image=image_original_pil,
        gt_pose=adjusted_pose,
        new_objects=objects_vis,
        paint_new_objects=False,
        return_vis_handle=debug_render,
    )
    
    if rendered_image is not None:
        add_info_to_image(rendered_image, frame_idx, len(filtered_objects), color_path)
        frames.append((rendered_image * 255).astype(np.uint8))

    if is_final_frame:
        # Save the video
        save_video_from_frames(frames, exp_out_path, exp_suffix)



