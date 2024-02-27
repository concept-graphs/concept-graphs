

import json
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
    """ Convert a json object to a dictionary representation """
    # Ensure input is a list for uniform processing
    if not isinstance(input_cfg, list):
        input_cfg = [input_cfg]
    
    result = []  # Initialize the result list to hold our dictionaries
    
    for entry in input_cfg:
        entry_dict = {}  # Dictionary to store current entry's data
        for key, value in entry.items():
            # Replace escaped newline and tab characters in strings
            if isinstance(value, str):
                formatted_value = value.replace("\\n", "\n").replace("\\t", "\t")
            else:
                formatted_value = value
            # Add the key-value pair to the current entry dictionary
            entry_dict[key] = formatted_value
        # Append the current entry dictionary to the result list
        result.append(entry_dict)
    
    # Return the result in dictionary format if it's a single entry or list of dictionaries otherwise
    return result[0] if len(result) == 1 else result

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

def save_hydra_config(hydra_cfg, exp_out_path):
    with open(exp_out_path / "config_params.json", "w") as f:
        json.dump(cfg_to_dict(hydra_cfg), f, indent=2)