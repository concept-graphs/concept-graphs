

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
