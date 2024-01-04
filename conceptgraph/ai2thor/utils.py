from typing import Tuple, Dict, List, Set, Union, Any, Optional, Mapping, cast
import math

import numpy as np
from scipy.spatial.transform import Rotation

from ai2thor.controller import Controller

def parse_object_receptacle_mapping(controller: Controller):
    # Parse the receptacle-object relationships
    obj_list = controller.last_event.metadata["objects"]
    obj2receptacle = {}
    receptacle2obj = {}
    for obj in obj_list:
        obj_id = obj['objectId']
        parents = obj['parentReceptacles']
        if parents is None:
            continue
        
        if len(parents) > 1:
            # print("Warning: object {} has more than one parent receptacle".format(obj_id))
            while "Floor" in parents:
                parents.remove("Floor")
            if len(parents) > 1:
                print("Warning: object {} has more than one parent receptacle".format(obj_id))
                # import pdb; pdb.set_trace()
                
        if len(parents) == 1:
            parent = parents[0]
            obj2receptacle[obj_id] = parent
            
            if parent not in receptacle2obj:
                receptacle2obj[parent] = []
            receptacle2obj[parent].append(obj_id)
            
    return obj2receptacle, receptacle2obj

def compute_position_dist(
    p0: Mapping[str, Any],
    p1: Mapping[str, Any],
    ignore_y: bool = False,
    l1_dist: bool = False,
) -> float:
    """Distance between two points of the form {"x": x, "y":y, "z":z"}."""
    if l1_dist:
        return (
            abs(p0["x"] - p1["x"])
            + (0 if ignore_y else abs(p0["y"] - p1["y"]))
            + abs(p0["z"] - p1["z"])
        )
    else:
        return math.sqrt(
            (p0["x"] - p1["x"]) ** 2
            + (0 if ignore_y else (p0["y"] - p1["y"]) ** 2)
            + (p0["z"] - p1["z"]) ** 2
        )

def compute_rotation_dist(a: Dict[str, float], b: Dict[str, float]):
    """Distance between rotations."""

    def deg_dist(d0: float, d1: float):
        dist = (d0 - d1) % 360
        return min(dist, 360 - dist)

    return sum(deg_dist(a[k], b[k]) for k in ["x", "y", "z"])

def compute_angle_between_rotations(a: Dict[str, float], b: Dict[str, float]):
    return np.abs(
        (180 / (2 * math.pi))
        * (
            Rotation.from_euler("xyz", [a[k] for k in "xyz"], degrees=True)
            * Rotation.from_euler("xyz", [b[k] for k in "xyz"], degrees=True).inv()
        ).as_rotvec()
    ).sum()