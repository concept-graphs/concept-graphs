import gzip
import json
import logging
import os
from pathlib import Path
import pickle
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

def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()

def to_tensor(numpy_array, device=None):
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if device is None:
        return torch.from_numpy(numpy_array)
    else:
        return torch.from_numpy(numpy_array).to(device)

def to_scalar(d: np.ndarray | torch.Tensor | float) -> int | float:
    '''
    Convert the d to a scalar
    '''
    if isinstance(d, float):
        return d
    
    elif "numpy" in str(type(d)):
        assert d.size == 1
        return d.item()
    
    elif isinstance(d, torch.Tensor):
        assert d.numel() == 1
        return d.item()
    
    else:
        raise TypeError(f"Invalid type for conversion: {type(d)}")

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