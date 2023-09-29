import argparse
import math
import os, glob
from pathlib import Path
import imageio
from matplotlib import pyplot as plt
import natsort
import gzip, pickle
from enum import Enum

from PIL import Image
from tqdm import trange
import open3d as o3d
import numpy as np
import time
import torch
import cv2

from supervision.detection.core import Detections
from supervision.draw.color import Color, ColorPalette

from conceptgraph.dataset.datasets_common import get_dataset

from conceptgraph.utils.general_utils import to_numpy
from conceptgraph.utils.vis import vis_result_fast, vis_result_slow_caption
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.utils import filter_gobs

def get_parser():
    parser = argparse.ArgumentParser(description="Visualize a series of point clouds as an animation.")
    parser.add_argument("--input_folder", type=str, help="Folder where the objects of the mapping process are stored.")
    return parser

def main(args):
    meta_path = os.path.join(args.input_folder, "meta.pkl.gz")
    frame_paths = glob.glob(os.path.join(args.input_folder, "*.pkl.gz"))
    frame_paths = [path for path in frame_paths if path != meta_path]
    frame_paths = natsort.natsorted(frame_paths)

    with gzip.open(meta_path, "rb") as f:
        meta_info = pickle.load(f)
        
    cfg = meta_info["cfg"]
    class_names = meta_info["class_names"]
    class_colors = meta_info["class_colors"]
    
    color_mode_instance = "class_agnostic" in cfg and cfg.class_agnostic
    
    video_save_path = Path(args.input_folder).parent / f"{cfg.gsa_variant}_{cfg.save_suffix}.mp4"

    # Save each video separately in this folder
    sep_video_save_folder = Path(args.input_folder).parent.parent / "anime_all_frames" / f"{cfg.gsa_variant}_{cfg.save_suffix}"
    os.makedirs(sep_video_save_folder, exist_ok=True)
    
    # The sv annotation module use BGR color and expects color to be in [0, 255]
    class_name_to_sv_color = {
        n: Color(round(c[2]*255), round(c[1]*255), round(c[0]*255))
        for n, c in zip(class_names, class_colors.values())
    }
    
    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        device="cpu",
        dtype=torch.float,
    )
    cam_K = dataset.get_cam_K()
    
    color_path = dataset.color_paths[0]
    image_pil = Image.open(color_path)
    
    # Create the visualization
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name = "Mapping",
        width = image_pil.width,
        height = image_pil.height,
    )
    
    view_ctrl = vis.get_view_control()
    view_ctrl.change_field_of_view(10)
    camera_param = view_ctrl.convert_to_pinhole_camera_parameters()
    
    main.show_bg = True
    
    result_frames = []
    result_frames_sep = {
        "image_rgb": [],
        "image_class": [],
        "render_rgb": [],
        "render_class": [],
    }
    
    i = 0
    for i in trange(len(frame_paths)):
        color_path = dataset.color_paths[i]
        image_pil = Image.open(color_path)
        image_rgb = np.array(image_pil)
        
        color_path = Path(color_path)
        detections_path = color_path.parent.parent / cfg.detection_folder_name / color_path.name
        detections_path = detections_path.with_suffix(".pkl.gz")
        color_path = str(color_path)
        detections_path = str(detections_path)
        
        # Load the detection results at this frame
        with gzip.open(detections_path, "rb") as f:
            gobs = pickle.load(f)
            
        # Filter out the objects as performed during the mapping process
        gobs = filter_gobs(cfg, gobs, image_rgb)
        
        # Annotate the image with the remaining detection results
        detections = Detections(
            xyxy = gobs['xyxy'],
            confidence = gobs['confidence'],
            class_id = gobs['class_id'],
            mask = gobs['mask'],
        )
        
        frame_class_names = gobs['classes']
        frame_classid_to_svcolor = [
            class_name_to_sv_color[class_name]
            for class_name in frame_class_names
        ]
        frame_color_palette = ColorPalette(frame_classid_to_svcolor)
        if color_mode_instance:
            image_class, labels = vis_result_fast(
                image_rgb, detections, frame_class_names,
                instance_random_color=True,
                draw_bbox=False,
            )
        else:
            image_class, labels = vis_result_fast(
                image_rgb, detections, frame_class_names, frame_color_palette,
                instance_random_color=False,
            )

        # Load the mapping results up to this frame
        with gzip.open(frame_paths[i], "rb") as f:
            frame = pickle.load(f)
            
        if i > 0:
            vis.clear_geometries()
            
        camera_pose = frame["camera_pose"]
        
        objects = MapObjectList()
        objects.load_serializable(frame["objects"])
        
        bg_objects = None
        if main.show_bg:
            if frame['bg_objects'] is None:
                # print("No background objects found.")
                pass
            else:
                bg_objects = MapObjectList()
                bg_objects.load_serializable(frame["bg_objects"])

        # First render the objects in RGB color
        pcds = objects.get_values("pcd")
        bboxes = objects.get_values("bbox")
        for geom in pcds + bboxes:
            vis.add_geometry(geom, reset_bounding_box = i<2)
        
        if bg_objects is not None:
            for geom in bg_objects.get_values("pcd"):
                vis.add_geometry(geom, reset_bounding_box = i<2)
        
        camera_param.extrinsic = np.linalg.inv(to_numpy(camera_pose))
        view_ctrl.convert_from_pinhole_camera_parameters(camera_param)
        view_ctrl.camera_local_translate(forward=-0.4, right=0.0, up=0.0)
        vis.poll_events()
        vis.update_renderer()
        
        render_rgb = vis.capture_screen_float_buffer(False)
        render_rgb = np.asarray(render_rgb)
        
        # Then render the objects in class-coded colors
        if color_mode_instance:
            objects.color_by_instance()
            if bg_objects is not None:
                bg_objects.color_by_instance()
        else:
            objects.color_by_most_common_classes(class_colors, color_bbox=True)
            if bg_objects is not None:
                bg_objects.color_by_most_common_classes(class_colors, color_bbox=True)
            
        pcds = objects.get_values("pcd")
        bboxes = objects.get_values("bbox")
        for geom in pcds + bboxes:
            vis.add_geometry(geom, reset_bounding_box = i<2)
        
        if bg_objects is not None:
            for geom in bg_objects.get_values("pcd"):
                vis.add_geometry(geom, reset_bounding_box = i<2)
        
        camera_param.extrinsic = np.linalg.inv(to_numpy(camera_pose))
        view_ctrl.convert_from_pinhole_camera_parameters(camera_param)
        view_ctrl.camera_local_translate(forward=-0.4, right=0.0, up=0.0)
        vis.poll_events()
        vis.update_renderer()
        
        render_class = vis.capture_screen_float_buffer(False)
        render_class = np.asarray(render_class)
        
        # plt.subplot(2, 2, 1)
        # plt.imshow(image_pil)
        # plt.subplot(2, 2, 2)
        # plt.imshow(image_class)
        # plt.subplot(2, 2, 3)
        # plt.imshow(render_rgb)
        # plt.subplot(2, 2, 4)
        # plt.imshow(render_class)
        # plt.show()
        
        result_frames_sep["image_rgb"].append(image_rgb)
        result_frames_sep["image_class"].append(image_class)
        result_frames_sep["render_rgb"].append((render_rgb * 255).astype(np.uint8))
        result_frames_sep["render_class"].append((render_class * 255).astype(np.uint8))
        
        # Stack the two renders side-by-side, on the bottom of the image and annotated image
        image_stack = np.concatenate([image_rgb, image_class], axis=1)
        render_stack = np.concatenate([render_rgb, render_class], axis=1)
        render_stack = (render_stack * 255).astype(np.uint8)
        vis_stack = np.concatenate([image_stack, render_stack], axis=0)
        
        # plt.figure(figsize=(20, 10))
        # plt.imshow(vis_stack)
        # plt.axis("off")
        # plt.show()
        
        result_frames.append(vis_stack)
        
    vis.destroy_window()
    
    # Save the result as a video
    imageio.mimwrite(video_save_path, result_frames, fps=10)
    
    for k, v in result_frames_sep.items():
        sep_video_save_path = sep_video_save_folder / f"{k}.mp4"
        imageio.mimwrite(sep_video_save_path, v, fps=10)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)