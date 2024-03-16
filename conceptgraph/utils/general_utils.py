import json
from pathlib import Path
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

def get_exp_out_path(dataset_root, scene_id, exp_suffix):
    exp_out_path = Path(dataset_root) / scene_id / "exps" / f"{exp_suffix}"
    exp_out_path.mkdir(exist_ok=True, parents=True)
    return exp_out_path

def get_vis_out_path(exp_out_path):
    vis_folder_path = exp_out_path / "vis"
    vis_folder_path.mkdir(exist_ok=True, parents=True)
    return vis_folder_path

def get_det_out_path(exp_out_path):
    detections_folder_path = exp_out_path / "detections"
    detections_folder_path.mkdir(exist_ok=True, parents=True)
    return detections_folder_path


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
