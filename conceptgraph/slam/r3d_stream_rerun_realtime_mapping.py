'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# Standard library imports
from typing import Mapping
import uuid
from conceptgraph.utils.optional_rerun_wrapper import OptionalReRun, orr_log_annotated_image, orr_log_camera, orr_log_depth_image, orr_log_edges, orr_log_objs_pcd_and_bbox, orr_log_rgb_image
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.geometry import rotation_matrix_to_quaternion
from conceptgraph.utils.logging_metrics import DenoisingTracker, MappingTracker
from conceptgraph.utils.record3d_utils import DemoApp
from conceptgraph.utils.vlm import get_obj_rel_from_image_gpt4v, get_openai_client
import cv2
import os
import PyQt5

# Set the QT_QPA_PLATFORM_PLUGIN_PATH environment variable
pyqt_plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt", "plugins", "platforms")
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = pyqt_plugin_path


import copy
# from line_profiler import profile
import os
from pathlib import Path
import gzip
import pickle

# Related third party imports
from PIL import Image

import numpy as np
# from open3d import io
from open3d.io import read_pinhole_camera_parameters
import torch
from tqdm import trange

import hydra
from omegaconf import DictConfig

# Local application/library specific imports
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import OnlineObjectRenderer, filter_detections, save_video_from_frames, vis_result_fast_on_depth, vis_result_for_vlm
from conceptgraph.utils.ious import (
    mask_subtract_contained
)
from conceptgraph.utils.general_utils import ObjectClasses, find_existing_image_path, get_det_out_path, get_exp_out_path, get_stream_data_out_path, load_saved_detections, load_saved_hydra_json_config, measure_time, save_detection_results, save_hydra_config, save_pointcloud, should_exit_early

from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
from conceptgraph.slam.utils import (
    filter_gobs,
    filter_objects_w_edges,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    denoise_objects,
    filter_objects,
    merge_objects, 
    detections_to_obj_pcd_and_bbox,
    prepare_objects_save_vis,
    process_cfg,
    process_pcd,
    processing_needed,
    resize_gobs,
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
    merge_obj_matches
)

# Detection utils
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.utils.vis import vis_result_fast
from conceptgraph.utils.general_utils import get_vis_out_path
from conceptgraph.utils.general_utils import cfg_to_dict
from conceptgraph.utils.general_utils import check_run_detections
from conceptgraph.utils.vis import save_video_detections

from ultralytics import YOLO
from ultralytics import SAM
import supervision as sv
import open_clip

# Disable torch gradient computation
torch.set_grad_enabled(False)

# A logger for this file
@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="rerun_realtime_mapping")
# @profile
def main(cfg : DictConfig):
    
    app = DemoApp()
    app.connect_to_device(dev_idx=0)
    
    tracker = MappingTracker()
    
    orr = OptionalReRun()
    orr.set_use_rerun(cfg.use_rerun)
    
    orr.init("realtime_mapping")
    orr.spawn()

    owandb = OptionalWandB()
    owandb.set_use_wandb(cfg.use_wandb)

    owandb.init(project="concept-graphs", 
            #    entity="concept-graphs",
                config=cfg_to_dict(cfg),
               )
    cfg = process_cfg(cfg)

    # # Initialize the dataset
    # dataset = get_dataset(
    #     dataconfig=cfg.dataset_config,
    #     start=cfg.start,
    #     end=cfg.end,
    #     stride=cfg.stride,
    #     basedir=cfg.dataset_root,
    #     sequence=cfg.scene_id,
    #     desired_height=cfg.image_height,
    #     desired_width=cfg.image_width,
    #     device="cpu",
    #     dtype=torch.float,
    # )
    # cam_K = dataset.get_cam_K()

    objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)

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
    det_exp_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.detections_exp_suffix, make_dir=False)

    # we need to make sure to use the same classes as the ones used in the detections
    detections_exp_cfg = cfg_to_dict(cfg)
    obj_classes = ObjectClasses(
        classes_file_path=detections_exp_cfg['classes_file'], 
        bg_classes=detections_exp_cfg['bg_classes'], 
        skip_bg=detections_exp_cfg['skip_bg']
    )

    # if we need to do detections
    run_detections = check_run_detections(cfg.force_detection, det_exp_path)
    det_exp_pkl_path = get_det_out_path(det_exp_path)
    det_exp_vis_path = get_vis_out_path(det_exp_path)
    
    stream_rgb_path, stream_depth_path, stream_poses_path = get_stream_data_out_path(cfg.dataset_root, cfg.scene_id)
    
    prev_adjusted_pose = None

    if run_detections:
        print("\n".join(["Running detections..."] * 10))
        det_exp_path.mkdir(parents=True, exist_ok=True)

        ## Initialize the detection models
        detection_model = measure_time(YOLO)('yolov8l-world.pt')
        sam_predictor = SAM('sam_l.pt') # SAM('mobile_sam.pt') # UltraLytics SAM
        # sam_predictor = measure_time(get_sam_predictor)(cfg) # Normal SAM
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        clip_model = clip_model.to(cfg.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # Set the classes for the detection model
        detection_model.set_classes(obj_classes.get_classes_arr())

        openai_client = get_openai_client()
        
    else:
        print("\n".join(["NOT Running detections..."] * 10))

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    exit_early_flag = False
    counter = 0
    frame_idx = 0
    total_frames = 500 # adjust as you like
    for frame_idx in trange(total_frames):
        k=1
        tracker.curr_frame_idx = frame_idx
        counter+=1
        orr.set_time_sequence("frame", frame_idx)

        # Check if we should exit early only if the flag hasn't been set yet
        if not exit_early_flag and should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True

        # If exit early flag is set and we're not at the last frame, skip this iteration
        if exit_early_flag and frame_idx < total_frames - 1:
            continue
        
        s_rgb, s_depth, s_intrinsic_mat, s_camera_pose = app.get_frame_data()

        # save the rgb to the stream folder with an appropriate name
        curr_stream_rgb_path = stream_rgb_path / f"{frame_idx}.jpg"
        cv2.imwrite(str(curr_stream_rgb_path), s_rgb)
        color_path = curr_stream_rgb_path
        k=1
        
        if cfg.save_detections:
        
            # save depth to the stream folder with an appropriate name
            curr_stream_depth_path = stream_depth_path / f"{frame_idx}.png"
            cv2.imwrite(str(curr_stream_depth_path), s_depth)
            
            # save the camera pose to the stream folder with an appropriate name 
            curr_stream_pose_path = stream_poses_path / f"{frame_idx}.npz"
            np.savez(str(curr_stream_pose_path), s_camera_pose)

        # Read info about current frame from dataset
        # color image
        # color_path = Path(dataset.color_paths[frame_idx])
        image_original_pil = Image.open(color_path)
        # color and depth tensors, and camera instrinsics matrix
        # color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]
        
        k=1

        # d_color_tensor, d_depth_tensor, d_intrinsics, *d_ = dataset[frame_idx]
        color_tensor = torch.from_numpy(s_rgb.astype('float32')) 
        depth_tensor = torch.from_numpy(s_depth.astype('float32'))
        intrinsics = s_intrinsic_mat

        
        
        k=1
        
        # Covert to numpy and do some sanity checks
        # depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy() # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        # Load image detections for the current frame
        raw_gobs = None
        gobs = None # stands for grounded observations
        detections_path = det_exp_pkl_path / (color_path.stem + ".pkl.gz")
        if run_detections:
            results = None
            # opencv can't read Path objects...
            image = cv2.imread(str(color_path)) # This will in BGR color space
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Do initial object detection
            results = detection_model.predict(color_path, conf=0.1, verbose=False)
            confidences = results[0].boxes.conf.cpu().numpy()
            detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
            xyxy_tensor = results[0].boxes.xyxy
            xyxy_np = xyxy_tensor.cpu().numpy()

            # Get Masks Using SAM or MobileSAM
            # UltraLytics SAM
            sam_out = sam_predictor.predict(color_path, bboxes=xyxy_tensor, verbose=False)
            masks_tensor = sam_out[0].masks.data

            masks_np = masks_tensor.cpu().numpy()

            # Create a detections object that we will save later
            curr_det = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=detection_class_ids,
                mask=masks_np,
            )
            
            # First, filter the detections
            filtered_detections, labels = filter_detections(
                detections=curr_det, 
                classes=obj_classes.get_classes_arr(),
                top_x_detections=15,
                confidence_threshold=0.1,
                given_labels = detection_class_labels
            )

            ### You can turn on the VLM detections if you'd like
            ### They're just still a little too slow for the streaming use case
            # Then, use the filtered detections and labels to annotate the image
            # annotated_image_for_vlm, labels = vis_result_for_vlm(
            #     image=image, 
            #     detections=filtered_detections,  # Use filtered detections here
            #     labels=labels,  # Use the labels obtained from filtering
            #     draw_bbox=True,
            #     thickness=5,
            #     text_scale=1,
            #     text_thickness=2,
            #     text_padding=3,
            #     save_path=str(det_exp_vis_path / color_path.name).replace(".jpg", "_for_vlm_process")
            # )
            # vis_save_path_for_vlm = str((det_exp_vis_path / color_path.name).with_suffix(".jpg")).replace(".jpg", "_for_vlm.jpg")
            
            # cv2.imwrite(str(vis_save_path_for_vlm), annotated_image_for_vlm)
            # print(f"Line 313, vis_save_path_for_vlm: {vis_save_path_for_vlm}")
            # edges = get_obj_rel_from_image_gpt4v(openai_client, vis_save_path_for_vlm, labels)
            edges = [] # for now, we're not using the VLM detections
            l=1 
            # Compute and save the clip features of detections
            image_crops, image_feats, text_feats = compute_clip_features_batched(
                image_rgb, curr_det, clip_model, clip_preprocess, clip_tokenizer, obj_classes.get_classes_arr(), cfg.device)

            # increment total object detections
            tracker.increment_total_detections(len(curr_det.xyxy))

            # Save results
            # Convert the detections to a dict. The elements are in np.array
            results = {
                # add new uuid for each detection 
                "xyxy": curr_det.xyxy,
                "confidence": curr_det.confidence,
                "class_id": curr_det.class_id,
                "mask": curr_det.mask,
                "classes": obj_classes.get_classes_arr(),
                "image_crops": image_crops,
                "image_feats": image_feats,
                "text_feats": text_feats,
                "detection_class_labels": detection_class_labels,
                "labels": labels,
                "edges": edges,
            }

            raw_gobs = results

            # save the detections if needed
            if cfg.save_detections:

                vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg")
                # Visualize and save the annotated image
                annotated_image, labels = vis_result_fast(image, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path), annotated_image)

                depth_image_rgb = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                depth_image_rgb = depth_image_rgb.astype(np.uint8)
                depth_image_rgb = cv2.cvtColor(depth_image_rgb, cv2.COLOR_GRAY2BGR)
                annotated_depth_image, labels = vis_result_fast_on_depth(depth_image_rgb, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth.jpg"), annotated_depth_image)
                cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth_only.jpg"), depth_image_rgb)
                # curr_detection_name = (vis_save_path.stem + ".pkl.gz")
                # with gzip.open(det_exp_pkl_path / curr_detection_name , "wb") as f:
                #     pickle.dump(results, f)
                # /home/kuwajerw/new_local_data/new_record3d/ali_apartment/apt_scan_no_smooth_processed/exps/r_detections_stride1000_2/detections/0

                save_detection_results(det_exp_pkl_path / vis_save_path.stem, results)
        else:
            # load the detections
            # color_path = str(color_path)
            # detections_path = str(detections_path)
            # str(det_exp_pkl_path / color_path.stem)
            raw_gobs = load_saved_detections(det_exp_pkl_path / color_path.stem)
            # with gzip.open(detections_path, "rb") as f:
            #     raw_gobs = pickle.load(f)

        # get pose, this is the untrasformed pose.
        # d_unt_pose = dataset.poses[frame_idx]
        unt_pose = s_camera_pose
        # unt_pose = unt_pose.cpu().numpy()

        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose
        
        prev_adjusted_pose = orr_log_camera(intrinsics, adjusted_pose, prev_adjusted_pose, cfg.image_width, cfg.image_height, frame_idx)
        
        orr_log_rgb_image(color_path)
        
        orr_log_annotated_image(color_path, det_exp_vis_path)

        orr_log_depth_image(depth_tensor)

        # resize the observation if needed
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        len_resized_gobs = len(resized_gobs['mask'])
        temp_gobs = copy.deepcopy(resized_gobs)
        # filter the observations
        filtered_gobs = filter_gobs(resized_gobs, image_rgb, 
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )
        
        # check and print if any detections were filtered
        if len_resized_gobs != len(filtered_gobs['mask']):
            print(f"=========================Filtered {len_resized_gobs - len(filtered_gobs['mask'])} detections")
            k=1

        gobs = filtered_gobs

        if len(gobs['mask']) == 0: # no detections in this frame
            continue

        # this helps make sure things like pillows on couches are separate objects
        gobs['mask'] = mask_subtract_contained(gobs['xyxy'], gobs['mask'])

        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth_array,
            masks=gobs['mask'],
            cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
            image_rgb=image_rgb,
            trans_pose=adjusted_pose,
            min_points_threshold=cfg.min_points_threshold,
            spatial_sim_type=cfg.spatial_sim_type,
            obj_pcd_max_points=cfg.obj_pcd_max_points,
            device=cfg.device,
        )
        k=1

        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                )
                # obj["pcd"] = obj["pcd"].remove_radius_outlier(nb_points=20, radius=0.02)[0]
                # obj["pcd"] = obj["pcd"].remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)[0]
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=cfg['spatial_sim_type'], 
                    pcd=obj["pcd"],
                )

        detection_list = make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx
        )

        if len(detection_list) == 0: # no detections, skip
            continue

        # if no objects yet in the map,
        # just add all the objects from the current frame
        # then continue, no need to match or merge
        if len(objects) == 0:
            objects.extend(detection_list)
            tracker.increment_total_objects(len(detection_list))
            owandb.log({
                    "total_objects_so_far": tracker.get_total_objects(),
                    "objects_this_frame": len(detection_list),
                })
            continue 

        ## compute similarities and then merge
        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=cfg['spatial_sim_type'], 
            detection_list=detection_list, 
            objects=objects,
            downsample_voxel_size=cfg['downsample_voxel_size']
        )

        visual_sim = compute_visual_similarities(detection_list, objects)

        agg_sim = aggregate_similarities(
            match_method=cfg['match_method'], 
            phys_bias=cfg['phys_bias'], 
            spatial_sim=spatial_sim, 
            visual_sim=visual_sim
        )

        # Perform matching of detections to existing objects
        match_indices = match_detections_to_objects(
            agg_sim=agg_sim, 
            detection_threshold=cfg['sim_threshold']  # Use the sim_threshold from the configuration
        )
        print(f"Line 575, len(objects) before merge: {len(objects)}")
        initial_objects_count = len(objects)

        # Now merge the detected objects into the existing objects based on the match indices
        objects = merge_obj_matches(
            detection_list=detection_list, 
            objects=objects, 
            match_indices=match_indices,
            downsample_voxel_size=cfg['downsample_voxel_size'], 
            dbscan_remove_noise=cfg['dbscan_remove_noise'], 
            dbscan_eps=cfg['dbscan_eps'], 
            dbscan_min_points=cfg['dbscan_min_points'], 
            spatial_sim_type=cfg['spatial_sim_type'], 
            device=cfg['device']
            # Note: Removed 'match_method' and 'phys_bias' as they do not appear in the provided merge function
        )
        print(f"Line 575, len(objects) AFTER merge: {len(objects)}")



        # Step 1: Generate match_indices_w_new_obj with indices for new objects
        # Initial count of objects before processing new detections
        new_object_count = 0  # Counter for new objects

        # Create a list of match indices with new objects index instead of None
        match_indices_w_new_obj = []
        for match_index in match_indices:
            if match_index is None:
                # Assign the future index for new objects and increment the counter
                new_obj_index = initial_objects_count + new_object_count
                match_indices_w_new_obj.append(new_obj_index)
                new_object_count += 1
            else:
                match_indices_w_new_obj.append(match_index)

        # Step 2: Create a mapping from 2D detection labels to detection indices
        detection_label_to_index = {label: index for index, label in enumerate(gobs['detection_class_labels'])}
        
        # Step 3: Use match_indices_w_new_obj for translating 2D edges to indices in the existing objects list
        curr_edges_3d_by_index = []
        for edge in gobs['edges']:
            obj1_label, relation, obj2_label = edge
            
            # Find the 2D detection indices for obj1 and obj2 using the full label
            obj1_index = detection_label_to_index.get(obj1_label)  # Use the full label
            obj2_index = detection_label_to_index.get(obj2_label)  # Use the full label
            
            # check that the indices are not None
            if (obj1_index is None) or (obj2_index is None):
                print(f"Line 623, obj1_index: {obj1_index}")
                print(f"Line 623, obj2_index: {obj2_index}")
                print(f"Line 624, obj1_label: {obj1_label}")
                print(f"Line 624, obj2_label: {obj2_label}")
                k=1
                # sometimes gpt4v returns a relation with a class that is not in the detections
                continue
            
            
            # check that the object indices are not out of range
            if (obj1_index is None) or (obj1_index >= len(match_indices_w_new_obj)):
                continue
            if (obj2_index is None) or (obj2_index >= len(match_indices_w_new_obj)):
                continue
            
            # Directly map 2D detection indices to object list indices using match_indices_w_new_obj
            obj1_objects_index = match_indices_w_new_obj[obj1_index] if obj1_index is not None else None
            obj2_objects_index = match_indices_w_new_obj[obj2_index] if obj2_index is not None else None

            curr_edges_3d_by_index.append((obj1_objects_index, relation, obj2_objects_index))

        print(f"Line 624, curr_edges_3d_by_index: {curr_edges_3d_by_index}")
        
        # Add the new edges to the map
        for (obj_1_idx, rel_type, obj_2_idx) in curr_edges_3d_by_index:
            map_edges.add_or_update_edge(obj_1_idx, obj_2_idx, rel_type)
            
        # Just making a copy of the edges by object number for viz
        map_edges_by_curr_obj_num = []
        for (obj1_idx, obj2_idx), map_edge in map_edges.edges_by_index.items():
            # check if the idxes are more than the length of the objects, if so, continue
            if obj1_idx >= len(objects) or obj2_idx >= len(objects):
                continue
            obj1_curr_obj_num = objects[obj1_idx]['curr_obj_num']
            obj2_curr_obj_num = objects[obj2_idx]['curr_obj_num']
            rel_type = map_edge.rel_type
            map_edges_by_curr_obj_num.append((obj1_curr_obj_num, rel_type, obj2_curr_obj_num))

        is_final_frame = frame_idx == total_frames - 1
        if is_final_frame:
            print("Final frame detected. Performing final post-processing...")
            k=1

        ### Perform post-processing periodically if told so

        # Denoising
        if processing_needed(
            cfg["denoise_interval"],
            cfg["run_denoise_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            print(f"Line 675, Denoising objects for frame {frame_idx}")
            objects = measure_time(denoise_objects)(
                downsample_voxel_size=cfg['downsample_voxel_size'], 
                dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                dbscan_eps=cfg['dbscan_eps'], 
                dbscan_min_points=cfg['dbscan_min_points'], 
                spatial_sim_type=cfg['spatial_sim_type'], 
                device=cfg['device'], 
                objects=objects
            )

        # Filtering
        if processing_needed(
            cfg["filter_interval"],
            cfg["run_filter_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            print(f"Line 688, Filtering objects for frame {frame_idx}")
            # objects = filter_objects(
            objects = filter_objects_w_edges(
                obj_min_points=cfg['obj_min_points'], 
                obj_min_detections=cfg['obj_min_detections'], 
                objects=objects,
                map_edges=map_edges
            )

        # Merging
        if processing_needed(
            cfg["merge_interval"],
            cfg["run_merge_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            print(f"Line 699, Merging objects for frame {frame_idx}")
            objects = measure_time(merge_objects)(
                merge_overlap_thresh=cfg["merge_overlap_thresh"],
                merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                objects=objects,
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"],
                map_edges=map_edges
            )
            

        # what you need for the second log rerun function
        # for full loop, objects
        # for each object:
        # obj['pcd'], obj['bbox']       
            
        orr_log_objs_pcd_and_bbox(objects, obj_classes)
        
        # orr_log_edges(objects, map_edges, obj_classes) # not using edges for now 

        # Save the objects for the current frame, if needed
        if cfg.save_objects_all_frames:
            # Define the path for saving the current frame's objects
            save_path = obj_all_frames_out_path / f"{frame_idx:06d}.pkl.gz"

            # Filter objects based on minimum number of detections and prepare them for saving
            filtered_objects = [obj for obj in objects if obj['num_detections'] >= cfg.obj_min_detections]
            prepared_objects = prepare_objects_save_vis(MapObjectList(filtered_objects))

            # Create the result dictionary with camera pose and prepared objects
            result = { "camera_pose": adjusted_pose, "objects": prepared_objects}
            # also save the current frame_idx, num objects, and color path in results
            result["frame_idx"] = frame_idx
            result["num_objects"] = len(filtered_objects)
            result["color_path"] = str(color_path)
            # Save the result dictionary to a compressed file
            with gzip.open(save_path, 'wb') as f:
                pickle.dump(result, f)

        # Render the image with the filtered and colored objects
        if cfg.vis_render:
            # Initialize an empty list for objects meeting the criteria
            filtered_objects = [
                copy.deepcopy(obj) for obj in objects 
                if obj['num_detections'] >= cfg.obj_min_detections and not obj['is_background']
            ]
            objects_vis = MapObjectList(filtered_objects)

            # Apply coloring based on the configuration
            if cfg.class_agnostic:
                objects_vis.color_by_instance()
            else:
                objects_vis.color_by_most_common_classes(obj_classes)

            # Render the image with the filtered and colored objects
            rendered_image, vis = obj_renderer.step(
                image=image_original_pil,
                gt_pose=adjusted_pose,
                new_objects=objects_vis,
                paint_new_objects=False,
                return_vis_handle=cfg.debug_render,
            )

            # If debug mode is enabled, run the visualization

            # Convert the rendered image to uint8 format, if it exists
            if rendered_image is not None:
                rendered_image = (rendered_image * 255).astype(np.uint8)

                # Define text to be added to the image
                frame_info_text = f"Frame: {frame_idx}, Objects: {len(objects)}, Path: {str(color_path)}"

                # Set the font, size, color, and thickness of the text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color = (255, 0, 0)  # Blue in BGR
                thickness = 1
                line_type = cv2.LINE_AA

                # Get text size for positioning
                text_size, _ = cv2.getTextSize(frame_info_text, font, font_scale, thickness)

                # Set position for the text (bottom-left corner)
                position = (10, rendered_image.shape[0] - 10)  # 10 pixels from the bottom-left corner

                # Add the text to the image
                cv2.putText(rendered_image, frame_info_text, position, font, font_scale, color, thickness, line_type)

                frames.append(rendered_image)

            if is_final_frame:
                # Save frames as a mp4 video
                frames = np.stack(frames)
                video_save_path = exp_out_path / (f"s_mapping_{cfg.exp_suffix}.mp4")
                save_video_from_frames(frames, video_save_path, fps=10)
                print("Save video to %s" % video_save_path)

        owandb.log({
            "frame_idx": frame_idx,
            "counter": counter,
            "exit_early_flag": exit_early_flag,
            "is_final_frame": is_final_frame,
        })

        tracker.increment_total_objects(len(objects))
        tracker.increment_total_detections(len(detection_list))
        owandb.log({
                "total_objects": tracker.get_total_objects(),
                "objects_this_frame": len(objects),
                "total_detections": tracker.get_total_detections(),
                "detections_this_frame": len(detection_list),
                "frame_idx": frame_idx,
                "counter": counter,
                "exit_early_flag": exit_early_flag,
                "is_final_frame": is_final_frame,
                })
        # print("hey")
    # LOOP OVER -----------------------------------------------------
    
    # Save the rerun output if needed
    if cfg.use_rerun and cfg.save_rerun:
        temp = exp_out_path / f"rerun_{cfg.exp_suffix}.rrd"
        print("Mapping done!")
        print("If you want to save the rerun file, you should do so from the rerun viewer now.")
        print("You can't yet both save and log a file in rerun.")
        print("If you do, make a pull request!")
        print("Also, close the viewer before continuing, it frees up a lot of RAM, which helps for saving the pointclouds.")
        print(f"Feel free to copy and use this path below, or choose your own:\n{temp}")
        input(f"Then press Enter to continue.")
        k=1

    # Save the pointcloud
    if cfg.save_pcd:
        save_pointcloud(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            cfg=cfg,
            objects=objects,
            obj_classes=obj_classes,
            latest_pcd_filepath=cfg.latest_pcd_filepath,
            create_symlink=True,
            edges=map_edges
        )

    # Save metadata if all frames are saved
    if cfg.save_objects_all_frames:
        save_meta_path = obj_all_frames_out_path / f"meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump({
                'cfg': cfg,
                'class_names': obj_classes.get_classes_arr(),
                'class_colors': obj_classes.get_class_color_dict_by_index(),
            }, f)

    if run_detections:
        if cfg.save_video:
            save_video_detections(det_exp_path)

    owandb.finish()

if __name__ == "__main__":
    main()
