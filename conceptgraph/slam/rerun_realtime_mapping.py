'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# Standard library imports
from typing import Mapping
import uuid
from conceptgraph.utils.optional_rerun_wrapper import OptionalReRun
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.geometry import rotation_matrix_to_quaternion
from conceptgraph.utils.logging_metrics import DenoisingTracker, MappingTracker
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
from conceptgraph.utils.vis import OnlineObjectRenderer, save_video_from_frames, vis_result_fast_on_depth
from conceptgraph.utils.ious import (
    mask_subtract_contained
)
from conceptgraph.utils.general_utils import ObjectClasses, find_existing_image_path, get_det_out_path, get_exp_out_path, load_saved_detections, load_saved_hydra_json_config, measure_time, save_detection_results, save_hydra_config, save_pointcloud, should_exit_early

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.utils import (
    filter_gobs,
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

import rerun as rr

# Disable torch gradient computation
torch.set_grad_enabled(False)

# A logger for this file
@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="rerun_realtime_mapping")
# @profile
def main(cfg : DictConfig):
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
    else:
        print("\n".join(["NOT Running detections..."] * 10))

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    exit_early_flag = False
    counter = 0
    for frame_idx in trange(len(dataset)):
        tracker.curr_frame_idx = frame_idx
        counter+=1
        orr.set_time_sequence("frame", frame_idx)

        # Check if we should exit early only if the flag hasn't been set yet
        if not exit_early_flag and should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True

        # If exit early flag is set and we're not at the last frame, skip this iteration
        if exit_early_flag and frame_idx < len(dataset) - 1:
            continue

        # Read info about current frame from dataset
        # color image
        color_path = Path(dataset.color_paths[frame_idx])
        image_original_pil = Image.open(color_path)
        # color and depth tensors, and camera instrinsics matrix
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]
        

        # Covert to numpy and do some sanity checks
        depth_tensor = depth_tensor[..., 0]
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
        unt_pose = dataset.poses[frame_idx]
        unt_pose = unt_pose.cpu().numpy()

        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose
        
        
        # Extract intrinsic parameters
        focal_length = [intrinsics[0, 0].item(), intrinsics[1, 1].item()]  # Assuming fx = fy, else use the average
        principal_point = [intrinsics[0, 2].item(), intrinsics[1, 2].item()]
        resolution = [image_rgb.shape[1], image_rgb.shape[0]]  # Width x Height from the RGB image
        
        # Initialize and log camera intrinsics
        orr.log(
            "world/camera",
            orr.Pinhole(
                resolution=resolution,
                focal_length=focal_length,
                principal_point=principal_point,
            ),
        )

        # Log RGB image
        orr.log(
            "world/camera/rgb_image_encoded",
            orr.ImageEncoded(path=color_path)
        )
        
        # check if vis path is a file that exists
        base_vis_save_path = det_exp_vis_path / color_path.stem
        existing_vis_save_path = find_existing_image_path(base_vis_save_path, ['.jpg', '.png'])
        
        if existing_vis_save_path:
            # Log the visualization image
            orr.log(
                "world/camera/rgb_image_annotated",
                orr.ImageEncoded(path=existing_vis_save_path)
            )
        
        
        # Convert adjusted_pose to translation and quaternion
        translation = adjusted_pose[:3, 3].tolist()  # Last column
        rotation_matrix = adjusted_pose[:3, :3]
        quaternion = orr.Quaternion(xyzw=rotation_matrix_to_quaternion(rotation_matrix))
        orr.log(
            "world/camera", orr.Transform3D(translation=translation, rotation=quaternion, from_parent=False)
        )

        # Assuming depth_tensor is already available and in millimeters
        # Convert depth_tensor from millimeters to meters for rerun (if necessary)
        # This conversion depends on your data's units; adjust accordingly
        depth_in_meters = depth_tensor.numpy() #  / 1000.0  # Convert mm to meters if necessary

        # Ensure depth data is in the expected format for rerun (HxW)
        # depth_in_meters should be a 2D numpy array at this point
        assert len(depth_in_meters.shape) == 2, "Depth data must be a 2D array"

        # Log the depth image using the previously logged Pinhole camera model
        # Specify meter parameter based on your depth data's units
        orr.log(
            "world/camera/depth",
            orr.DepthImage(depth_in_meters , meter=0.9999999)  # Use meter=1.0 if depth_in_meters is already in meters
        )

        # resize the observation if needed
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        # filter the observations
        filtered_gobs = filter_gobs(resized_gobs, image_rgb, 
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )

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

        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                )
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

        is_final_frame = frame_idx == len(dataset) - 1

        ### Perform post-processing periodically if told so

        # Denoising
        if processing_needed(
            cfg["denoise_interval"],
            cfg["run_denoise_final_frame"],
            frame_idx,
            is_final_frame,
        ):
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
            objects = filter_objects(
                obj_min_points=cfg['obj_min_points'], 
                obj_min_detections=cfg['obj_min_detections'], 
                objects=objects
            )

        # Merging
        if processing_needed(
            cfg["merge_interval"],
            cfg["run_merge_final_frame"],
            frame_idx,
            is_final_frame,
        ):
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
            )
            
        # Assume 'objects' is your list of objects for the current frame
        for obj in objects:
            # tracker.total_object_count
            # print(f"Line 525, tracker.total_object_count: {tracker.total_object_count}")
            obj_label = f"{obj['curr_obj_num']}_{obj['class_name']}"
            # obj['new_counter']
            # print(f"-------------------------------------------------Line 528, obj['new_counter']: {obj['new_counter']}")
            # print(f"Line 527, obj['curr_obj_num']: {obj['curr_obj_num']}")
            # replace whatspace with underscore in the label
            obj_label = obj_label.replace(" ", "_")
            # print(f"Line 527, obj_label: {obj_label}")
            # print(f"Line 530, obj['num_obj_in_class']: {obj['num_obj_in_class']}")
            # print(f"Line 529, obj['id']: {obj['id']}")
            # if obj_label in tracker.prev_obj_names:
            # tracker.prev_obj_names
            entity_path = f"world/objects/{obj_label}"

            # Convert points and colors to NumPy arrays
            positions = np.asarray(obj['pcd'].points)
            if hasattr(obj['pcd'], 'colors') and len(obj['pcd'].colors) > 0:
                colors = np.asarray(obj['pcd'].colors) * 255
                # make them ints
                colors = colors.astype(np.uint8)
            else:
                colors = None

            # Log point cloud data
            orr.log(entity_path + "/pcd", orr.Points3D(positions, colors=colors))

            # Assuming bbox is extracted as before
            bbox = obj['bbox']
            centers = [bbox.center]
            half_sizes = [bbox.extent /2 ]
            # Convert rotation matrix to quaternion
            bbox_quaternion = [rotation_matrix_to_quaternion(bbox.R)]

            orr.log(entity_path + "/bbox", orr.Boxes3D(centers=centers, half_sizes=half_sizes, rotations=bbox_quaternion))

            # Handle class_id which might be a list or an int
            class_id = obj['class_id'][0] if isinstance(obj['class_id'], list) else obj['class_id']

            annotation_color = [int(c * 255) for c in obj['inst_color']]  # Normalize to [0, 255]
            k=1

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

        if counter % 10 == 0:
            # save the pointcloud
            save_pointcloud(
                exp_suffix=cfg.exp_suffix,
                exp_out_path=exp_out_path,
                cfg=cfg,
                objects=objects,
                obj_classes=obj_classes,
                latest_pcd_filepath=cfg.latest_pcd_filepath,
                create_symlink=True
            )

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

    # Save the pointcloud
    if cfg.save_pcd:
        save_pointcloud(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            cfg=cfg,
            objects=objects,
            obj_classes=obj_classes,
            latest_pcd_filepath=cfg.latest_pcd_filepath,
            create_symlink=True
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
