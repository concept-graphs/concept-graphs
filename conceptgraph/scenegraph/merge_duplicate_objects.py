"""
Post-process the results from the SLAM to merge duplicate objects and filter out
objects with too few points or detections.
"""

import argparse
import copy
import os
import pickle
import gzip

import matplotlib
import numpy as np

from conceptgraph.slam.slam_classes import MapObjectList

from conceptgraph.slam.utils import filter_objects, merge_objects

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--savefile", type=str, required=True)
    
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
        
    cfg = results['cfg']
    class_names = results['class_names']
    class_colors = results['class_colors']
        
    objects = MapObjectList()
    objects.load_serializable(results['objects'])
    
    # Run the post-processing filtering and merging in instructed to do so
    cfg.obj_min_points = args.obj_min_points
    cfg.obj_min_detections = args.obj_min_detections
    cfg.merge_overlap_thresh = args.merge_overlap_thresh
    cfg.merge_visual_sim_thresh = args.merge_visual_sim_thresh
    cfg.merge_text_sim_thresh = args.merge_text_sim_thresh
    objects = filter_objects(cfg, objects)
    objects = merge_objects(cfg, objects)
    
    # if not args.no_clip:
    #     print("Initializing CLIP model...")
    #     clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    #     clip_model = clip_model.to("cuda")
    #     clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    #     print("Done initializing CLIP model.")

    cmap = matplotlib.colormaps.get_cmap("turbo")
    
    # Handle the background objects if they exist
    print(results.keys())
    bg_objects = None
    if 'bg_objects' in results and results['bg_objects'] is not None:
        bg_objects = MapObjectList()
        bg_objects.load_serializable(results['bg_objects'])

        pcds_bg = copy.deepcopy(bg_objects.get_values("pcd"))
        bboxes_bg = copy.deepcopy(bg_objects.get_values("bbox"))
        
        indices_bg = np.arange(len(objects), len(objects) + len(bg_objects))
        
        objects.extend(bg_objects)
    
    # pcds = copy.deepcopy(objects.get_values("pcd"))
    # bboxes = copy.deepcopy(objects.get_values("bbox"))

    # # similarities = objects.compute_similarities(text_query_ft)
    # objects_clip_fts = objects.get_stacked_values_torch("clip_ft")

    objects = objects.to_serializable()
    # print(len(objects))
    
    print(f"\nStarted with {len(results['objects'])} objects then filtered to {len(objects)} objects\n")

    # Save the results
    # If the parent directory doesn't exist, create it
    if not os.path.exists(os.path.dirname(args.savefile)):
        os.makedirs(os.path.dirname(args.savefile), exist_ok=True)
    with gzip.open(args.savefile, "wb") as f:
        pickle.dump(objects, f)
    