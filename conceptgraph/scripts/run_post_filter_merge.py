import copy
import os
import pickle
import gzip
import argparse

import matplotlib
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import open_clip
from gradslam.structures.pointclouds import Pointclouds

from conceptgraph.slam.slam_classes import MapObjectList

from conceptgraph.slam.utils import filter_objects, merge_objects

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--rgb_pcd_path", type=str, default=None)
    
    # To inspect the results of merge_overlap_objects
    # This is mainly to quickly try out different thresholds
    parser.add_argument("--merge_overlap_thresh", type=float, default=-1)
    parser.add_argument("--merge_visual_sim_thresh", type=float, default=-1)
    parser.add_argument("--merge_text_sim_thresh", type=float, default=-1)
    parser.add_argument("--obj_min_points", type=int, default=0)
    parser.add_argument("--obj_min_detections", type=int, default=0)
    
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    result_path = args.result_path

    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)
        
    objects = MapObjectList()
    objects.load_serializable(results['objects'])
    
    # Run the post-processing filtering and merging in instructed to do so
    cfg = copy.deepcopy(results['cfg'])
    cfg.obj_min_points = args.obj_min_points
    cfg.obj_min_detections = args.obj_min_detections
    cfg.merge_overlap_thresh = args.merge_overlap_thresh
    cfg.merge_visual_sim_thresh = args.merge_visual_sim_thresh
    cfg.merge_text_sim_thresh = args.merge_text_sim_thresh
    objects = filter_objects(cfg, objects)
    objects = merge_objects(cfg, objects)
    
    updated_results = {
        'objects': objects.to_serializable(),
        'cfg': results['cfg'],
        'class_names': results['class_names'],
        'class_colors': results['class_colors'],
    }
    
    if 'bg_objects' in results and results['bg_objects'] is not None:
        updated_results['bg_objects'] = results['bg_objects']
    
    save_path = result_path[:-7] + "_post.pkl.gz"
    
    with gzip.open(save_path, "wb") as f:
        pickle.dump(updated_results, f)
    print(f"Saved full point cloud to {save_path}")
    