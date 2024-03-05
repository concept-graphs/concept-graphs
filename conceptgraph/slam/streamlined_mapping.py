'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# Standard library imports
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import gzip
import pickle

# Related third party imports
from PIL import Image
import cv2
import imageio
import numpy as np
import open3d as o3d
# from open3d import io
from open3d.io import read_pinhole_camera_parameters
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf

# Local application/library specific imports
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import OnlineObjectRenderer, save_video_from_frames
from conceptgraph.utils.ious import (
    compute_2d_box_contained_batch
)
from conceptgraph.utils.general_utils import ObjectClasses, get_det_out_path, get_exp_config_save_path, get_exp_out_path, load_saved_hydra_json_config, save_hydra_config, to_tensor

from conceptgraph.slam.slam_classes import MapObjectList, DetectionList
from conceptgraph.slam.utils import (
    create_or_load_colors,
    filter_gobs,
    merge_obj2_into_obj1, 
    denoise_objects,
    filter_objects,
    merge_objects, 
    gobs_to_detection_list,
    resize_gobs,
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects
)

# BG_CLASSES = ["wall", "floor", "ceiling"]

# Disable torch gradient computation
torch.set_grad_enabled(False)

def compute_match_batch(cfg, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    Compute object association based on spatial and visual similarities
    
    Args:
        spatial_sim: a MxN tensor of spatial similarities
        visual_sim: a MxN tensor of visual similarities
    Returns:
        A MxN tensor of binary values, indicating whether a detection is associate with an object. 
        Each row has at most one 1, indicating one detection can be associated with at most one existing object.
        One existing object can receive multiple new detections
    '''
    assign_mat = torch.zeros_like(spatial_sim)
    if cfg.match_method == "sim_sum":
        sims = (1 + cfg.phys_bias) * spatial_sim + (1 - cfg.phys_bias) * visual_sim # (M, N)
        row_max, row_argmax = torch.max(sims, dim=1) # (M,), (M,)
        for i in row_max.argsort(descending=True):
            if row_max[i] > cfg.sim_threshold:
                assign_mat[i, row_argmax[i]] = 1
            else:
                break
    else:
        raise ValueError(f"Unknown matching method: {cfg.match_method}")
    
    return assign_mat

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

@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="streamlined_mapping")
def main(cfg : DictConfig):
    cfg = process_cfg(cfg)

    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )
    # cam_K = dataset.get_cam_K()




    objects = MapObjectList(device=cfg.device)

    # if not cfg.skip_bg:
    #     # Handle the background detection separately
    #     # Each class of them are fused into the map as a single object
    #     bg_objects = {
    #         c: None for c in BG_CLASSES
    #     }
    # else:
    #     bg_objects = None



    # For visualization
    if cfg.vis_render:
        view_param = read_pinhole_camera_parameters(cfg.render_camera_path)
        obj_renderer = OnlineObjectRenderer(
            view_param = view_param,
            base_objects = None, 
            gray_map = False,
        )
        frames = []

    # output folder for this mapping experiment
    exp_out_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.exp_suffix)
    
    # output folder of the detections experiment to use
    det_exp_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.detections_exp_suffix)
    # the actual folder with the detections from the detections experiment
    detections_folder = get_det_out_path(det_exp_path)
    
    # we need to make sure to use the same classes as the ones used in the detections
    detections_exp_cfg = load_saved_hydra_json_config(det_exp_path)
    obj_classes = ObjectClasses(
        classes_file_path=detections_exp_cfg['classes_file'], 
        bg_classes=detections_exp_cfg['bg_classes'], 
        skip_bg=detections_exp_cfg['skip_bg']
    )
    k=1
    # cfg.detections_config = detections_exp_cfg
    # OmegaConf.update(cfg, 'detections_config', detections_exp_cfg, merge=False)
    # cfg['detections_config'] = detections_exp_cfg
    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)
    
    if not cfg.skip_bg:
        bg_objects = {
            c: None for c in obj_classes.get_bg_classes_arr()
        }
    else:    
        bg_objects = None
    # detections_skip_bg = 
    
    # obj_classes = ObjectClasses(
    #     cfg.classes_file, bg_classes=cfg.bg_classes, skip_bg=cfg.skip_bg
    # )


    
    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    for idx in trange(len(dataset)):

        # Read info about current frame from dataset
        # color image
        color_path = Path(dataset.color_paths[idx])
        image_original_pil = Image.open(color_path)
        # color and depth tensors, and camera instrinsics matrix
        color_tensor, depth_tensor, intrinsics, *_ = dataset[idx]
        
        # Covert to numpy and do some sanity checks
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy() # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        # Load image detections for the current frame
        gobs = None # stands for grounded SAM observations
        detections_path = detections_folder / (color_path.stem + ".pkl.gz")
        # detections_path = color_path.parent.parent / cfg.detection_folder_name / color_path.name
        # detections_path = detections_path.with_suffix(".pkl.gz")
        color_path = str(color_path)
        detections_path = str(detections_path)

        with gzip.open(detections_path, "rb") as f:
            gobs = pickle.load(f)

        k  = 1

        # depth_image = Image.open(depth_path)
        # depth_array = np.array(depth_image) / dataset.png_depth_scale
        # depth_tensor = torch.from_numpy(depth_array).float().to(cfg['device']).T

        # get pose, this is the untrasformed pose.
        unt_pose = dataset.poses[idx]
        unt_pose = unt_pose.cpu().numpy()

        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose

        # # Now we will load the detections list 
        # fg_detection_list = DetectionList()
        # bg_detection_list = DetectionList()
        
        # gobs = resize_gobs(gobs, image_rgb)
        # gobs = filter_gobs(gobs, image_rgb, , )


        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            image=image_rgb,
            depth_array=depth_array,
            cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
            idx=idx,
            gobs=gobs,
            trans_pose=adjusted_pose,
            class_names=obj_classes.get_classes_arr(),
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            skip_bg=cfg['skip_bg'],  # Assuming 'skip_bg' is a config parameter
            color_path=color_path,
            mask_area_threshold=cfg['mask_area_threshold'],
            max_bbox_area_ratio=cfg['max_bbox_area_ratio'],
            mask_conf_threshold=cfg['mask_conf_threshold'],
            min_points_threshold=cfg['min_points_threshold'],
            spatial_sim_type=cfg['spatial_sim_type'],
            # New configuration parameters passed explicitly
            downsample_voxel_size=cfg['downsample_voxel_size'],
            dbscan_remove_noise=cfg['dbscan_remove_noise'],
            dbscan_eps=cfg['dbscan_eps'],
            dbscan_min_points=cfg['dbscan_min_points']
        )


        if len(bg_detection_list) > 0:
            for detected_object in bg_detection_list:
                class_name = detected_object['class_name'][0]
                if bg_objects[class_name] is None:
                    bg_objects[class_name] = detected_object
                else:
                    matched_obj = bg_objects[class_name]
                    matched_det = detected_object
                    # bg_objects[class_name] = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=False)
                    bg_objects[class_name] = merge_obj2_into_obj1(
                        obj1=matched_obj, 
                        obj2=matched_det, 
                        downsample_voxel_size=cfg['downsample_voxel_size'], 
                        dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                        dbscan_eps=cfg['dbscan_eps'], 
                        dbscan_min_points=cfg['dbscan_min_points'], 
                        spatial_sim_type=cfg['spatial_sim_type'], 
                        device=cfg['device'], 
                        run_dbscan=False
                    )

        if len(fg_detection_list) == 0:
            continue



        if len(objects) == 0:
            # Add all detections to the map
            for i in range(len(fg_detection_list)):
                objects.append(fg_detection_list[i])

            # Skip the similarity computation
            continue

        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=cfg['spatial_sim_type'], 
            detection_list=fg_detection_list, 
            objects=objects,
            downsample_voxel_size=cfg['downsample_voxel_size']
        )

        visual_sim = compute_visual_similarities(fg_detection_list, objects)

        agg_sim = aggregate_similarities(
            match_method=cfg['match_method'], 
            phys_bias=cfg['phys_bias'], 
            spatial_sim=spatial_sim, 
            visual_sim=visual_sim
        )



        # Threshold sims according to cfg. Set to negative infinity if below threshold
        agg_sim[agg_sim < cfg.sim_threshold] = float('-inf')

        objects = merge_detections_to_objects(
            downsample_voxel_size=cfg['downsample_voxel_size'], 
            dbscan_remove_noise=cfg['dbscan_remove_noise'], 
            dbscan_eps=cfg['dbscan_eps'], 
            dbscan_min_points=cfg['dbscan_min_points'], 
            spatial_sim_type=cfg['spatial_sim_type'], 
            device=cfg['device'], 
            match_method=cfg['match_method'], 
            phys_bias=cfg['phys_bias'], 
            detection_list=fg_detection_list, 
            objects=objects, 
            agg_sim=agg_sim
        )


        # Perform post-processing periodically if told so
        if cfg['denoise_interval'] > 0 and (idx+1) % cfg['denoise_interval'] == 0:
            objects = denoise_objects(
                downsample_voxel_size=cfg['downsample_voxel_size'], 
                dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                dbscan_eps=cfg['dbscan_eps'], 
                dbscan_min_points=cfg['dbscan_min_points'], 
                spatial_sim_type=cfg['spatial_sim_type'], 
                device=cfg['device'], 
                objects=objects
            )
        if cfg.filter_interval > 0 and (idx+1) % cfg.filter_interval == 0:
            objects = filter_objects(
                obj_min_points=cfg['obj_min_points'], 
                obj_min_detections=cfg['obj_min_detections'], 
                objects=objects
            )
        if cfg['merge_interval'] > 0 and (idx+1) % cfg['merge_interval'] == 0:
            objects = merge_objects(
                merge_overlap_thresh=cfg['merge_overlap_thresh'], 
                merge_visual_sim_thresh=cfg['merge_visual_sim_thresh'], 
                merge_text_sim_thresh=cfg['merge_text_sim_thresh'],
                objects=objects, 
                downsample_voxel_size=cfg['downsample_voxel_size'], 
                dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                dbscan_eps=cfg['dbscan_eps'], 
                dbscan_min_points=cfg['dbscan_min_points'], 
                spatial_sim_type=cfg['spatial_sim_type'], 
                device=cfg['device']
            )

        if cfg.save_objects_all_frames:
            save_all_path = obj_all_frames_out_path / f"{idx:06d}.pkl.gz"
            objects_to_save = MapObjectList([
                _ for _ in objects if _['num_detections'] >= cfg.obj_min_detections
            ])

            objects_to_save = prepare_objects_save_vis(objects_to_save)

            if (not cfg.skip_bg) and (len(bg_detection_list) > 0): # i.e. background objects are being used
                bg_objects_to_save = MapObjectList([_ for _ in bg_objects.values() if _ is not None])
                bg_objects_to_save = prepare_objects_save_vis(bg_objects_to_save)
            else:
                bg_objects_to_save = None

            result = {
                "camera_pose": adjusted_pose,
                "objects": objects_to_save,
                "bg_objects": bg_objects_to_save,
            }
            with gzip.open(save_all_path, 'wb') as f:
                pickle.dump(result, f)

        if cfg.vis_render:
            objects_vis = MapObjectList([
                copy.deepcopy(_) for _ in objects if _['num_detections'] >= cfg.obj_min_detections
            ])

            if cfg.class_agnostic:
                objects_vis.color_by_instance()
            else:
                objects_vis.color_by_most_common_classes(obj_classes)

            rendered_image, vis = obj_renderer.step(
                image = image_original_pil,
                gt_pose = adjusted_pose,
                new_objects = objects_vis,
                paint_new_objects=False,
                return_vis_handle = cfg.debug_render,
            )

            if cfg.debug_render:
                vis.run()
                del vis

            # Convert to uint8
            if rendered_image is not None:
                rendered_image = (rendered_image * 255).astype(np.uint8)
                frames.append(rendered_image)

        # print(
        #     f"Finished image {idx} of {len(dataset)}",
        #     f"Now we have {len(objects)} objects.",
        #     f"Effective objects {len([_ for _ in objects if _['num_detections'] >= cfg.obj_min_detections])}"
        # )

    if bg_objects is not None:
        bg_objects = MapObjectList([_ for _ in bg_objects.values() if _ is not None])
        # bg_objects = denoise_objects(cfg, bg_objects)
        bg_objects = denoise_objects(
                downsample_voxel_size=cfg['downsample_voxel_size'], 
                dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                dbscan_eps=cfg['dbscan_eps'], 
                dbscan_min_points=cfg['dbscan_min_points'], 
                spatial_sim_type=cfg['spatial_sim_type'], 
                device=cfg['device'], 
                objects=bg_objects
            )

    # objects = denoise_objects(cfg, objects)
    objects = denoise_objects(
                downsample_voxel_size=cfg['downsample_voxel_size'], 
                dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                dbscan_eps=cfg['dbscan_eps'], 
                dbscan_min_points=cfg['dbscan_min_points'], 
                spatial_sim_type=cfg['spatial_sim_type'], 
                device=cfg['device'], 
                objects=objects
            )

    # Save the full point cloud before post-processing
    if cfg.save_pcd:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {
            'objects': objects.to_serializable(),
            'bg_objects': None if bg_objects is None else bg_objects.to_serializable(),
            'cfg': cfg,
            'class_names': obj_classes.get_classes_arr(),
            'class_colors': obj_classes.class_to_color,
        }

        # pcd_save_path = cfg.dataset_root / \
        #     cfg.scene_id / 'pcd_saves' / f"full_pcd_{cfg.gsa_variant}_{cfg.save_suffix}.pkl.gz"
        pcd_save_path = exp_out_path / f"full_pcd_{cfg.exp_suffix}.pkl.gz"
        # make the directory if it doesn't exist
        pcd_save_path.parent.mkdir(parents=True, exist_ok=True)
        pcd_save_path = str(pcd_save_path)

        with gzip.open(pcd_save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved full point cloud to {pcd_save_path}")

    objects = filter_objects(
        obj_min_points=cfg['obj_min_points'], 
        obj_min_detections=cfg['obj_min_detections'], 
        objects=objects
    )
    objects = merge_objects(
        merge_overlap_thresh=cfg['merge_overlap_thresh'], 
        merge_visual_sim_thresh=cfg['merge_visual_sim_thresh'], 
        merge_text_sim_thresh=cfg['merge_text_sim_thresh'],
        objects=objects, 
        downsample_voxel_size=cfg['downsample_voxel_size'], 
        dbscan_remove_noise=cfg['dbscan_remove_noise'], 
        dbscan_eps=cfg['dbscan_eps'], 
        dbscan_min_points=cfg['dbscan_min_points'], 
        spatial_sim_type=cfg['spatial_sim_type'], 
        device=cfg['device']
    )

    # Save again the full point cloud after the post-processing
    if cfg.save_pcd:
        results['objects'] = objects.to_serializable()
        pcd_save_path = pcd_save_path[:-7] + "_post.pkl.gz"
        with gzip.open(pcd_save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved full point cloud after post-processing to {pcd_save_path}")

    if cfg.save_objects_all_frames:
        save_meta_path = obj_all_frames_out_path / f"meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump({
                'cfg': cfg,
                'class_names': obj_classes.get_classes_arr(),
                'class_colors': obj_classes.class_to_color,
            }, f)

    if cfg.vis_render:
        # Still render a frame after the post-processing
        objects_vis = MapObjectList([
            _ for _ in objects if _['num_detections'] >= cfg.obj_min_detections
        ])

        if cfg.class_agnostic:
            objects_vis.color_by_instance()
        else:
            objects_vis.color_by_most_common_classes(obj_classes)

        rendered_image, vis = obj_renderer.step(
            image = image_original_pil,
            gt_pose = adjusted_pose,
            new_objects = objects_vis,
            paint_new_objects=False,
            return_vis_handle = False,
        )

        # Convert to uint8
        rendered_image = (rendered_image * 255).astype(np.uint8)
        frames.append(rendered_image)

        # Save frames as a mp4 video
        frames = np.stack(frames)
        video_save_path = exp_out_path / (f"s_mapping_{cfg.exp_suffix}.mp4")
        # video_save_path = (
        #     cfg.dataset_root
        #     / cfg.scene_id
        #     / ("objects_mapping-%s-%s.mp4" % (cfg.gsa_variant, cfg.save_suffix))
        # )
        # imageio.mimwrite(video_save_path, frames, fps=10)
        save_video_from_frames(frames, video_save_path, fps=10)
        print("Save video to %s" % video_save_path)

if __name__ == "__main__":
    main()
