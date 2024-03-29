'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# Standard library imports
from typing import Mapping
import uuid
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

import wandb

import hydra
from omegaconf import DictConfig

# Local application/library specific imports
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import OnlineObjectRenderer, save_video_from_frames, vis_result_fast_on_depth
from conceptgraph.utils.ious import (
    mask_subtract_contained
)
from conceptgraph.utils.general_utils import ObjectClasses, get_det_out_path, get_exp_out_path, load_saved_detections, load_saved_hydra_json_config, measure_time, save_detection_results, save_hydra_config, save_pointcloud, should_exit_early

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

# Disable torch gradient computation
torch.set_grad_enabled(False)

import open3d.visualization.gui as gui
import open3d as o3d
import threading
import time

CLOUD_NAME = "points"

@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="gui_realtime_mapping")
def main(cfg : DictConfig):
    tracker = MappingTracker()
    
    wandb.init(project="concept-graphs", 
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

    # # For visualization
    # if cfg.vis_render:
    #     view_param = read_pinhole_camera_parameters(cfg.render_camera_path)
    #     obj_renderer = OnlineObjectRenderer(
    #         view_param = view_param,
    #         base_objects = None, 
    #         gray_map = False,
    #     )
    #     frames = []
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

    det_exp_vis_path = None
    obj_all_frames_out_path = None
    det_exp_vis_path = None
    detection_model = None
    sam_predictor = None
    clip_model = None
    clip_preprocess = None
    clip_tokenizer = None
    if run_detections:
        det_exp_path.mkdir(parents=True, exist_ok=True)

        det_exp_vis_path = get_vis_out_path(det_exp_path)

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

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    exit_early_flag = False
    counter = 0
    
    MultiWinApp(
        cfg=cfg, 
        dataset=dataset, 
        objects=objects,
        exp_out_path=exp_out_path,
        det_exp_path=det_exp_path,
        detections_exp_cfg=detections_exp_cfg,
        obj_classes=obj_classes,
        run_detections=run_detections,
        det_exp_pkl_path=det_exp_pkl_path,
        det_exp_vis_path=det_exp_vis_path,
        detection_model=detection_model,
        sam_predictor=sam_predictor,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_tokenizer=clip_tokenizer,
        obj_all_frames_out_path=obj_all_frames_out_path,
        tracker=tracker,
        counter=counter,
        exit_early_flag=exit_early_flag,
    ).run()


class MultiWinApp:

    def __init__(
        self,
        cfg,
        dataset,
        objects,
        exp_out_path,
        det_exp_path,
        detections_exp_cfg,
        obj_classes,
        run_detections,
        det_exp_pkl_path,
        det_exp_vis_path,
        detection_model,
        sam_predictor,
        clip_model,
        clip_preprocess,
        clip_tokenizer,
        obj_all_frames_out_path,
        tracker,
        counter,
        exit_early_flag,
    ):
        self.is_done = False
        self.n_snapshots = 0
        self.cloud = None
        self.main_vis = None
        self.snapshot_pos = None
        self.is_paused = False
        self.frame_idx = 0
        self.curr_obj_num = 0
        self.prev_obj_names = []
        self.prev_bbox_names = []

        self.cfg = cfg
        self.dataset = dataset
        self.objects = objects
        self.exp_out_path = exp_out_path
        self.det_exp_path = det_exp_path
        self.detections_exp_cfg = detections_exp_cfg
        self.obj_classes = obj_classes
        self.run_detections = run_detections
        self.det_exp_pkl_path = det_exp_pkl_path
        self.det_exp_vis_path = det_exp_vis_path
        self.detection_model = detection_model
        self.sam_predictor = sam_predictor
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        self.obj_all_frames_out_path = obj_all_frames_out_path
        self.tracker = tracker
        self.counter = counter
        self.exit_early_flag = exit_early_flag
        
        self.rgb_images = []
        self.depth_images = []
        self.curr_image_rgb = None
        self.curr_depth_array = None
        self.load_images()  # Load RGB and depth images
        
        # em = self.window.theme.font_size
        # em = 2.0
        # margin = 0.5 * em
        # self.panel = o3d.visualization.gui.Vert(0.5 * em, o3d.visualization.gui.Margins(margin))
        # self.panel.add_child(o3d.visualization.gui.Label("Color image"))
        # self.rgb_widget = o3d.visualization.gui.ImageWidget(self.rgb_images[0])
        # self.panel.add_child(self.rgb_widget)
        # self.panel.add_child(o3d.visualization.gui.Label("Depth image (normalized)"))
        # self.depth_widget = o3d.visualization.gui.ImageWidget(self.depth_images[0])
        # self.panel.add_child(self.depth_widget)
        # self.window.add_child(self.panel)

    def toggle_pause(self):  # Define the toggle method
        self.is_paused = not self.is_paused
        if self.is_paused:
            print("Updates paused.")
        else:
            print("Updates resumed.")
            
    def load_images(self):
        # Dummy loading function - replace with actual loading logic
        rgbd_data = o3d.data.SampleRedwoodRGBDImages()
        for color_path, depth_path in zip(rgbd_data.color_paths, rgbd_data.depth_paths):
            self.rgb_images.append(o3d.io.read_image(color_path))
            self.depth_images.append(o3d.io.read_image(depth_path))

    # def setup_image_display_window(self):

        
    def update_images(self):
        # Convert numpy images to Open3D images and update widgets
        o3d_rgb_image = o3d.geometry.Image(self.curr_image_rgb)
        o3d_depth_image = o3d.geometry.Image((self.curr_depth_array / self.curr_depth_array.max() * 255).astype(np.uint8))  # Example normalization

        self.rgb_widget.update_image(o3d_rgb_image)
        self.depth_widget.update_image(o3d_depth_image)
            
    def run(self):
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        self.main_vis = o3d.visualization.O3DVisualizer(
            "Open3D - Multi-Window Demo")
        self.main_vis.add_action("Take snapshot in new window",
                                 self.on_snapshot)
        self.main_vis.add_action("Pause/Resume updates", lambda vis: self.toggle_pause())
        self.main_vis.set_on_close(self.on_main_window_closing)
        
        app.add_window(self.main_vis)
        # app.add_window(self.images_window)
        
        # # Setup the secondary window for images
        # # self.setup_image_display_window()
        # # self.image_window = app.create_window("RGB and Depth Images", 640, 480)
        # self.image_window = gui.Application.instance.create_window("RGB and Depth Images", 640, 480)
        # # self.image_window.create_window()
        # # self.image_window = o3d.visualization.Visualizer("RGB and Depth Images", 640, 480)
        # self.layout = o3d.visualization.gui.Vert(0, o3d.visualization.gui.Margins(10))
        # self.image_window.add_child(self.layout)

        # # Create image widgets
        # self.rgb_widget = o3d.visualization.gui.ImageWidget()
        # self.depth_widget = o3d.visualization.gui.ImageWidget()

        # # Add image widgets to the layout
        # self.layout.add_child(self.rgb_widget)
        # self.layout.add_child(self.depth_widget)
        
        
        
        self.snapshot_pos = (self.main_vis.os_frame.x, self.main_vis.os_frame.y)

        threading.Thread(target=self.update_thread).start()

        app.run()

    def on_snapshot(self, vis):
        self.n_snapshots += 1
        self.snapshot_pos = (self.snapshot_pos[0] + 50,
                             self.snapshot_pos[1] + 50)
        title = "Open3D - Multi-Window Demo (Snapshot #" + str(
            self.n_snapshots) + ")"
        new_vis = o3d.visualization.O3DVisualizer(title)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        new_vis.add_geometry(CLOUD_NAME + " #" + str(self.n_snapshots),
                             self.cloud, mat)
        new_vis.reset_camera_to_default()
        bounds = self.cloud.get_axis_aligned_bounding_box()
        extent = bounds.get_extent()
        new_vis.setup_camera(60, bounds.get_center(),
                             bounds.get_center() + [0, 0, -3], [0, -1, 0])
        o3d.visualization.gui.Application.instance.add_window(new_vis)
        new_vis.os_frame = o3d.visualization.gui.Rect(self.snapshot_pos[0],
                                                      self.snapshot_pos[1],
                                                      new_vis.os_frame.width,
                                                      new_vis.os_frame.height)


    def update_thread(self):
        # This is NOT the UI thread, need to call post_to_main_thread() to update
        # the scene or any part of the UI.
        pcd_data = o3d.data.DemoICPPointClouds()
        self.cloud = o3d.io.read_point_cloud(pcd_data.paths[0])
        bounds = self.cloud.get_axis_aligned_bounding_box()
        extent = bounds.get_extent()

        # pcd_data = None
        # self.cloud = None
        # bounds = None
        # extent = None

        # def add_first_cloud():
        #     # mat = o3d.visualization.rendering.MaterialRecord()
        #     # mat.shader = "defaultUnlit"
        #     # self.main_vis.add_geometry(CLOUD_NAME, self.cloud, mat)
        #     self.main_vis.add_geometry(CLOUD_NAME, self.cloud)
        #     self.main_vis.reset_camera_to_default()
        #     self.main_vis.setup_camera(60, bounds.get_center(),
        #                                bounds.get_center() + [0, 0, -3],
        #                                [0, -1, 0])

        # o3d.visualization.gui.Application.instance.post_to_main_thread(
        #     self.main_vis, add_first_cloud)

        while not self.is_done:
            time.sleep(0.1)

            if self.is_paused:  # Check if updates are paused
                continue
            
            # Check if we should exit early only if the flag hasn't been set yet
            if not self.exit_early_flag and should_exit_early(self.cfg.exit_early_file):
                print("Exit early signal detected. Skipping to the final frame...")
                self.exit_early_flag = True

            # If exit early flag is set and we're not at the last frame, skip this iteration
            if self.exit_early_flag and self.frame_idx < len(self.dataset) - 1:
                continue

            # Read info about current frame from dataset
            # color image
            color_path = Path(self.dataset.color_paths[self.frame_idx])
            image_original_pil = Image.open(color_path)
            # color and depth tensors, and camera instrinsics matrix
            color_tensor, depth_tensor, intrinsics, *_ = self.dataset[self.frame_idx]

            # Covert to numpy and do some sanity checks
            depth_tensor = depth_tensor[..., 0]
            depth_array = depth_tensor.cpu().numpy()
            color_np = color_tensor.cpu().numpy() # (H, W, 3)
            image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
            self.curr_image_rgb = image_rgb
            self.curr_depth_array = depth_array
            # o3d.visualization.gui.Application.instance.post_to_main_thread(
            #     self.image_window, self.update_images())
            assert image_rgb.max() > 1, "Image is not in range [0, 255]"

            # Load image detections for the current frame
            raw_gobs = None
            gobs = None # stands for grounded observations
            detections_path = self.det_exp_pkl_path / (color_path.stem + ".pkl.gz")
            if self.run_detections:
                results = None
                # opencv can't read Path objects...
                image = cv2.imread(str(color_path)) # This will in BGR color space
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Do initial object detection
                results = self.detection_model.predict(color_path, conf=0.1, verbose=False)
                confidences = results[0].boxes.conf.cpu().numpy()
                detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                xyxy_tensor = results[0].boxes.xyxy
                xyxy_np = xyxy_tensor.cpu().numpy()

                # Get Masks Using SAM or MobileSAM
                # UltraLytics SAM
                sam_out = self.sam_predictor.predict(color_path, bboxes=xyxy_tensor, verbose=False)
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
                    image_rgb, curr_det, self.clip_model, self.clip_preprocess, self.clip_tokenizer, self.obj_classes.get_classes_arr(), self.cfg.device)

                # increment total object detections
                self.tracker.increment_total_detections(len(curr_det.xyxy))

                # Save results
                # Convert the detections to a dict. The elements are in np.array
                results = {
                    # add new uuid for each detection 
                    "xyxy": curr_det.xyxy,
                    "confidence": curr_det.confidence,
                    "class_id": curr_det.class_id,
                    "mask": curr_det.mask,
                    "classes": self.obj_classes.get_classes_arr(),
                    "image_crops": image_crops,
                    "image_feats": image_feats,
                    "text_feats": text_feats,
                }

                raw_gobs = results
                
                # save the detections if needed
                if self.cfg.save_detections:

                    vis_save_path = (self.det_exp_vis_path / color_path.name).with_suffix(".jpg")
                    # Visualize and save the annotated image
                    annotated_image, labels = vis_result_fast(image, curr_det, self.obj_classes.get_classes_arr())
                    cv2.imwrite(str(vis_save_path), annotated_image)

                    depth_image_rgb = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                    depth_image_rgb = depth_image_rgb.astype(np.uint8)
                    depth_image_rgb = cv2.cvtColor(depth_image_rgb, cv2.COLOR_GRAY2BGR)
                    annotated_depth_image, labels = vis_result_fast_on_depth(depth_image_rgb, curr_det, self.obj_classes.get_classes_arr())
                    cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth.jpg"), annotated_depth_image)
                    cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth_only.jpg"), depth_image_rgb)
                    # curr_detection_name = (vis_save_path.stem + ".pkl.gz")
                    # with gzip.open(det_exp_pkl_path / curr_detection_name , "wb") as f:
                    #     pickle.dump(results, f)
                    # /home/kuwajerw/new_local_data/new_record3d/ali_apartment/apt_scan_no_smooth_processed/exps/r_detections_stride1000_2/detections/0

                    save_detection_results(self.det_exp_pkl_path / vis_save_path.stem, results)
            else:
                # load the detections
                # color_path = str(color_path)
                # detections_path = str(detections_path)
                # str(det_exp_pkl_path / color_path.stem)
                raw_gobs = load_saved_detections(self.det_exp_pkl_path / color_path.stem)
                # with gzip.open(detections_path, "rb") as f:
                #     raw_gobs = pickle.load(f)

            # get pose, this is the untrasformed pose.
            unt_pose = self.dataset.poses[self.frame_idx]
            unt_pose = unt_pose.cpu().numpy()
            k=1
            
            
            
            # Don't apply any transformation otherwise
            adjusted_pose = unt_pose

            # resize the observation if needed
            resized_gobs = resize_gobs(raw_gobs, image_rgb)
            # filter the observations
            filtered_gobs = filter_gobs(resized_gobs, image_rgb, 
                skip_bg=self.cfg.skip_bg,
                BG_CLASSES=self.obj_classes.get_bg_classes_arr(),
                mask_area_threshold=self.cfg.mask_area_threshold,
                max_bbox_area_ratio=self.cfg.max_bbox_area_ratio,
                mask_conf_threshold=self.cfg.mask_conf_threshold,
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
                min_points_threshold=self.cfg.min_points_threshold,
                spatial_sim_type=self.cfg.spatial_sim_type,
                obj_pcd_max_points=self.cfg.obj_pcd_max_points,
                device=self.cfg.device,
            )

            for obj in obj_pcds_and_bboxes:
                if obj:
                    obj["pcd"] = init_process_pcd(
                        pcd=obj["pcd"],
                        downsample_voxel_size=self.cfg["downsample_voxel_size"],
                        dbscan_remove_noise=self.cfg["dbscan_remove_noise"],
                        dbscan_eps=self.cfg["dbscan_eps"],
                        dbscan_min_points=self.cfg["dbscan_min_points"],
                    )
                    obj["bbox"] = get_bounding_box(
                        spatial_sim_type=self.cfg['spatial_sim_type'], 
                        pcd=obj["pcd"],
                    )

            detection_list = make_detection_list_from_pcd_and_gobs(
                obj_pcds_and_bboxes, gobs, color_path, self.obj_classes, self.frame_idx
            )

            if len(detection_list) == 0: # no detections, skip
                continue

            # if no objects yet in the map,
            # just add all the objects from the current frame
            # then continue, no need to match or merge
            if len(self.objects) == 0:
                self.objects.extend(detection_list)
                self.tracker.increment_total_objects(len(detection_list))
                wandb.log({
                        "total_objects_so_far": self.tracker.get_total_objects(),
                        "objects_this_frame": len(detection_list),
                    })
                continue 

            ### compute similarities and then merge
            spatial_sim = compute_spatial_similarities(
                spatial_sim_type=self.cfg['spatial_sim_type'], 
                detection_list=detection_list, 
                objects=self.objects,
                downsample_voxel_size=self.cfg['downsample_voxel_size']
            )

            visual_sim = compute_visual_similarities(detection_list, self.objects)

            agg_sim = aggregate_similarities(
                match_method=self.cfg['match_method'], 
                phys_bias=self.cfg['phys_bias'], 
                spatial_sim=spatial_sim, 
                visual_sim=visual_sim
            )

            # Perform matching of detections to existing objects
            match_indices = match_detections_to_objects(
                agg_sim=agg_sim, 
                detection_threshold=self.cfg['sim_threshold']  # Use the sim_threshold from the configuration
            )

            # Now merge the detected objects into the existing objects based on the match indices
            self.objects = merge_obj_matches(
                detection_list=detection_list, 
                objects=self.objects, 
                match_indices=match_indices,
                downsample_voxel_size=self.cfg['downsample_voxel_size'], 
                dbscan_remove_noise=self.cfg['dbscan_remove_noise'], 
                dbscan_eps=self.cfg['dbscan_eps'], 
                dbscan_min_points=self.cfg['dbscan_min_points'], 
                spatial_sim_type=self.cfg['spatial_sim_type'], 
                device=self.cfg['device']
                # Note: Removed 'match_method' and 'phys_bias' as they do not appear in the provided merge function
            )

            is_final_frame = self.frame_idx == len(self.dataset) - 1

            ### Perform post-processing periodically if told so

            # Denoising
            if processing_needed(
                self.cfg["denoise_interval"],
                self.cfg["run_denoise_final_frame"],
                self.frame_idx,
                is_final_frame,
            ):
                self.objects = measure_time(denoise_objects)(
                    downsample_voxel_size=self.cfg['downsample_voxel_size'], 
                    dbscan_remove_noise=self.cfg['dbscan_remove_noise'], 
                    dbscan_eps=self.cfg['dbscan_eps'], 
                    dbscan_min_points=self.cfg['dbscan_min_points'], 
                    spatial_sim_type=self.cfg['spatial_sim_type'], 
                    device=self.cfg['device'], 
                    objects=self.objects
                )

            # Filtering
            if processing_needed(
                self.cfg["filter_interval"],
                self.cfg["run_filter_final_frame"],
                self.frame_idx,
                is_final_frame,
            ):
                self.objects = filter_objects(
                    obj_min_points=self.cfg['obj_min_points'], 
                    obj_min_detections=self.cfg['obj_min_detections'], 
                    objects=self.objects
                )

            # Merging
            if processing_needed(
                self.cfg["merge_interval"],
                self.cfg["run_merge_final_frame"],
                self.frame_idx,
                is_final_frame,
            ):
                self.objects = measure_time(merge_objects)(
                    merge_overlap_thresh=self.cfg["merge_overlap_thresh"],
                    merge_visual_sim_thresh=self.cfg["merge_visual_sim_thresh"],
                    merge_text_sim_thresh=self.cfg["merge_text_sim_thresh"],
                    objects=self.objects,
                    downsample_voxel_size=self.cfg["downsample_voxel_size"],
                    dbscan_remove_noise=self.cfg["dbscan_remove_noise"],
                    dbscan_eps=self.cfg["dbscan_eps"],
                    dbscan_min_points=self.cfg["dbscan_min_points"],
                    spatial_sim_type=self.cfg["spatial_sim_type"],
                    device=self.cfg["device"],
                )

            # Save the objects for the current frame, if needed
            if self.cfg.save_objects_all_frames:
                # Define the path for saving the current frame's objects
                save_path = self.obj_all_frames_out_path / f"{self.frame_idx:06d}.pkl.gz"

                # Filter objects based on minimum number of detections and prepare them for saving
                filtered_objects = [obj for obj in self.objects if obj['num_detections'] >= self.cfg.obj_min_detections]
                prepared_objects = prepare_objects_save_vis(MapObjectList(filtered_objects))

                # Create the result dictionary with camera pose and prepared objects
                result = { "camera_pose": adjusted_pose, "objects": prepared_objects}
                # also save the current frame_idx, num objects, and color path in results
                result["frame_idx"] = self.frame_idx
                result["num_objects"] = len(filtered_objects)
                result["color_path"] = str(color_path)
                # Save the result dictionary to a compressed file
                with gzip.open(save_path, 'wb') as f:
                    pickle.dump(result, f)

            # Render the image with the filtered and colored objects
            if self.cfg.vis_render:
                # Initialize an empty list for objects meeting the criteria
                filtered_objects = [
                    copy.deepcopy(obj) for obj in self.objects 
                    if obj['num_detections'] >= self.cfg.obj_min_detections and not obj['is_background']
                ]
                objects_vis = MapObjectList(filtered_objects)

                # Apply coloring based on the configuration
                if self.cfg.class_agnostic:
                    objects_vis.color_by_instance()
                else:
                    objects_vis.color_by_most_common_classes(self.obj_classes)

                rendered_image = None
                # # Render the image with the filtered and colored objects
                # rendered_image, vis = obj_renderer.step(
                #     image=image_original_pil,
                #     gt_pose=adjusted_pose,
                #     new_objects=objects_vis,
                #     paint_new_objects=False,
                #     return_vis_handle=self.cfg.debug_render,
                # )

                # If debug mode is enabled, run the visualization

                # Convert the rendered image to uint8 format, if it exists
                # if rendered_image is not None:
                #     rendered_image = (rendered_image * 255).astype(np.uint8)

                #     # Define text to be added to the image
                #     frame_info_text = f"Frame: {self.frame_idx}, Objects: {len(self.objects)}, Path: {str(color_path)}"

                #     # Set the font, size, color, and thickness of the text
                #     font = cv2.FONT_HERSHEY_SIMPLEX
                #     font_scale = 0.5
                #     color = (255, 0, 0)  # Blue in BGR
                #     thickness = 1
                #     line_type = cv2.LINE_AA

                #     # Get text size for positioning
                #     text_size, _ = cv2.getTextSize(frame_info_text, font, font_scale, thickness)

                #     # Set position for the text (bottom-left corner)
                #     position = (10, rendered_image.shape[0] - 10)  # 10 pixels from the bottom-left corner

                #     # Add the text to the image
                #     cv2.putText(rendered_image, frame_info_text, position, font, font_scale, color, thickness, line_type)

                #     frames.append(rendered_image)

                # if is_final_frame:
                #     # Save frames as a mp4 video
                #     frames = np.stack(frames)
                #     video_save_path = self.exp_out_path / (f"s_mapping_{self.cfg.exp_suffix}.mp4")
                #     save_video_from_frames(frames, video_save_path, fps=10)
                #     print("Save video to %s" % video_save_path)

            if self.counter % 10 == 0:
                # save the pointcloud
                save_pointcloud(
                    exp_suffix=self.cfg.exp_suffix,
                    exp_out_path=self.exp_out_path,
                    cfg=self.cfg,
                    objects=self.objects,
                    obj_classes=self.obj_classes,
                    latest_pcd_filepath=self.cfg.latest_pcd_filepath,
                    create_symlink=True
                )

            wandb.log({
                "frame_idx": self.frame_idx,
                "counter": self.counter,
                "exit_early_flag": self.exit_early_flag,
                "is_final_frame": is_final_frame,
            })

            self.tracker.increment_total_objects(len(self.objects))
            self.tracker.increment_total_detections(len(detection_list))
            wandb.log({
                    "total_objects": self.tracker.get_total_objects(),
                    "objects_this_frame": len(self.objects),
                    "total_detections": self.tracker.get_total_detections(),
                    "detections_this_frame": len(detection_list),
                    "frame_idx": self.frame_idx,
                    "counter": self.counter,
                    "exit_early_flag": self.exit_early_flag,
                    "is_final_frame": is_final_frame,
                    })
            
            
            
            
            
            
            self.frame_idx += 1

            # all_points = np.vstack([np.asarray(obj['pcd'].points) for obj in self.objects])
            # all_colors = np.vstack([np.asarray(obj['pcd'].colors) for obj in self.objects])

            # combined_pcd = o3d.geometry.PointCloud()
            # combined_pcd.points = o3d.utility.Vector3dVector(all_points)
            # combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)

            # self.cloud.points = combined_pcd.points
            # self.cloud.colors = combined_pcd.colors
            
            # full_pcd_bounds = self.cloud.get_axis_aligned_bounding_box()
            
            def my_update_cloud():
                # Update points and colors
                # self.main_vis.remove_geometry("curr_points")
                # self.main_vis.add_geometry("curr_points", self.cloud)
                
                
                
                # Remove previous objects
                for obj_name in self.prev_obj_names:
                    self.main_vis.remove_geometry(obj_name)
                self.prev_obj_names = []
                
                # Remove previous bounding boxes
                for bbox_name in self.prev_bbox_names:
                    self.main_vis.remove_geometry(bbox_name)
                self.prev_bbox_names = []
                
                for obj_num, obj in enumerate(self.objects):
                    obj_label = f"{obj['curr_obj_num']}_{obj['class_name']}"
                    
                    obj_name = f"obj_{obj_label}"
                    bbox_name = f"bbox_{obj_label}"
                    
                    
                    
                    self.prev_obj_names.append(obj_name)
                    self.main_vis.add_geometry(obj_name, obj['pcd'])
                    
                    
                    self.prev_bbox_names.append(bbox_name)
                    self.main_vis.add_geometry(bbox_name, obj['bbox'] )
                    
                # if self.frame_idx == 1:    
                #     self.main_vis.reset_camera_to_default()
                #     self.main_vis.setup_camera(60, full_pcd_bounds.get_center(),
                #                             full_pcd_bounds.get_center() + [0, 0, -3],
                #                             [0, -1, 0])
                    
                    
            # pcd_data = o3d.data.DemoICPPointClouds()
            # self.cloud = o3d.io.read_point_cloud(pcd_data.paths[0])
            # bounds = self.cloud.get_axis_aligned_bounding_box()
            # extent = bounds.get_extent()
            # def add_first_cloud():
            #     # mat = o3d.visualization.rendering.MaterialRecord()
            #     # mat.shader = "defaultUnlit"
            #     # self.main_vis.add_geometry(CLOUD_NAME, self.cloud, mat)
            #     self.main_vis.add_geometry(CLOUD_NAME, self.cloud)
            #     self.main_vis.reset_camera_to_default()
            #     self.main_vis.setup_camera(60, bounds.get_center(),
            #                             bounds.get_center() + [0, 0, -3],
            #                             [0, -1, 0])

            # o3d.visualization.gui.Application.instance.post_to_main_thread(
            #     self.main_vis, add_first_cloud)
        

            if self.is_done:  # might have changed while sleeping
                break
            o3d.visualization.gui.Application.instance.post_to_main_thread(
                self.main_vis, my_update_cloud)
            
            
            

            
            ############# My thing is done #############################
            ############# My thing is done #############################
            ############# My thing is done #############################
            # Perturb the cloud with a random walk to simulate an actual read
            # k=1
            # pts = np.asarray(self.cloud.points)
            # magnitude = 0.005 * extent
            # displacement = magnitude * (np.random.random_sample(pts.shape) -
            #                             0.5)
            # new_pts = pts + displacement
            # self.cloud.points = o3d.utility.Vector3dVector(new_pts)

            
            # def update_cloud():
            #     # Note: if the number of points is less than or equal to the
            #     #       number of points in the original object that was added,
            #     #       using self.scene.update_geometry() will be faster.
            #     #       Requires that the point cloud be a t.PointCloud.
            #     self.main_vis.remove_geometry(CLOUD_NAME)
            #     mat = o3d.visualization.rendering.MaterialRecord()
            #     mat.shader = "defaultUnlit"
            #     self.main_vis.add_geometry(CLOUD_NAME, self.cloud, mat)

            # if self.is_done:  # might have changed while sleeping
            #     break
            # o3d.visualization.gui.Application.instance.post_to_main_thread(
            #     self.main_vis, update_cloud)
            
    def on_main_window_closing(self):
        self.is_done = True
        
        # Save the pointcloud
        if self.cfg.save_pcd:
            save_pointcloud(
                exp_suffix=self.cfg.exp_suffix,
                exp_out_path=self.exp_out_path,
                cfg=self.cfg,
                objects=self.objects,
                obj_classes=self.obj_classes,
                latest_pcd_filepath=self.cfg.latest_pcd_filepath,
                create_symlink=True
            )
            
        # Save metadata if all frames are saved
        if self.cfg.save_objects_all_frames:
            save_meta_path = self.obj_all_frames_out_path / f"meta.pkl.gz"
            with gzip.open(save_meta_path, "wb") as f:
                pickle.dump({
                    'cfg': self.cfg,
                    'class_names': self.obj_classes.get_classes_arr(),
                    'class_colors': self.obj_classes.get_class_color_dict_by_index(),
                }, f)

        if self.run_detections:
            if self.cfg.save_video:
                save_video_detections(self.det_exp_path)
            
        wandb.finish()
        return True  # False would cancel the close



if __name__ == "__main__":
    main()
