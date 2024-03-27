import cv2
import os
import PyQt5

# Set the QT_QPA_PLATFORM_PLUGIN_PATH environment variable
pyqt_plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt", "plugins", "platforms")
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = pyqt_plugin_path

import argparse
import json
import glob
import imageio
import natsort
import gzip, pickle
from enum import Enum

import open3d as o3d
import numpy as np
import time

from tqdm import trange

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.general_utils import to_numpy
from conceptgraph.utils.vis import better_camera_frustum, LineMesh

import distinctipy

class COLOR_MODE(Enum):
    RGB = 0
    CLASS = 1
    INSTANCE = 2
    
def create_ball_mesh(center, radius, color=(0, 1, 0)):
    """
    Create a colored mesh sphere.
    
    Args:
    - center (tuple): (x, y, z) coordinates for the center of the sphere.
    - radius (float): Radius of the sphere.
    - color (tuple): RGB values in the range [0, 1] for the color of the sphere.
    
    Returns:
    - o3d.geometry.TriangleMesh: Colored mesh sphere.
    """
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere

def get_parser():
    parser = argparse.ArgumentParser(description="Visualize a series of point clouds as an animation.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder where the objects of the mapping process are stored.")
    parser.add_argument("--meta_folder", type=str, default=None, help="Folder where the meta information is stored. Default the same as input_folder.")
    parser.add_argument("--edge_file", type=str, default=None, help="Path to the scene graph relationship json file. If not provided, the edges will not be shown.")
    parser.add_argument("--sleep_time", type=float, default=0.1, help="Time to sleep between each frame.")
    parser.add_argument("--follow_cam", action="store_true", help="If set, follow the camera pose.")
    parser.add_argument("--use_original_color", action="store_true", help="If set, will use color scheme from the CFSLAM pipeline.")
    parser.add_argument("--height_cutoff", type=float, default=np.inf, help="Object nodes above this height will not be shown.")
    return parser

cached_frame = None
def load_frame(path):
    global cached_frame
    if cached_frame is None:
        with gzip.open(path, "rb") as f:
            frame = pickle.load(f)
    else:
        frame = cached_frame
        
    if isinstance(frame, dict):
        camera_pose = frame.get("camera_pose")
        objects = MapObjectList()
        objects.load_serializable(frame.get("objects"))

        bg_objects = None
        if frame.get('bg_objects') is not None:
            bg_objects = MapObjectList()
            bg_objects.load_serializable(frame["bg_objects"])
    elif isinstance(frame, list):
        cached_frame = frame
        objects = MapObjectList()
        objects.load_serializable(frame)

        bg_objects = None
        camera_pose = None
    else:
        raise ValueError("Unknown frame type: ", type(frame))
    print()
    print("Loaded frame: ", path)
    print("Number of objects: ", len(objects))
    print("Frame_idx: ", frame.get("frame_idx"))
    print("Color_path: ", frame.get("color_path"))
    return camera_pose, objects, bg_objects

def main(args):
    # Load metadata
    if args.meta_folder is None:
        args.meta_folder = args.input_folder
        
    meta_path = os.path.join(args.meta_folder, "meta.pkl.gz")
    with gzip.open(meta_path, "rb") as f:
        meta_info = pickle.load(f)
        
    cfg = meta_info["cfg"]
    class_names = meta_info["class_names"]
    main.class_colors = meta_info["class_colors"]
    
    if not args.use_original_color:
        distinct_colors = distinctipy.get_colors(len(main.class_colors), pastel_factor=0.5)
        for k, c in zip(main.class_colors.keys(), distinct_colors):
            main.class_colors[k] = c
    
    # Scan the mapping results from all frames
    frame_paths = glob.glob(os.path.join(args.input_folder, "*.pkl.gz"))
    frame_paths = [path for path in frame_paths if path != meta_path]
    frame_paths = natsort.natsorted(frame_paths)
    
    # Load edge files
    if args.edge_file is not None:
        with open(args.edge_file, "r") as f:
            edges = json.load(f)

    # Create the visualization
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    view_ctrl = vis.get_view_control()
    camera_param = view_ctrl.convert_to_pinhole_camera_parameters()
    
    main.color_mode = COLOR_MODE.RGB
    if "class_agnostic" in cfg and cfg.class_agnostic:
        main.color_mode = COLOR_MODE.INSTANCE
        
    main.paused = False
    main.show_bg = True
    main.forward = True
    main.show_top = True
    main.show_graph = True
    main.show_cam = True
    main.show_bbox = True
    main.frame_idx = 0
    
    # Define the key callback functions
    def color_mode_rgb(vis):
        main.color_mode = COLOR_MODE.RGB
        
    def color_mode_class(vis):
        main.color_mode = COLOR_MODE.CLASS
        
    def color_mode_inst(vis):
        main.color_mode = COLOR_MODE.INSTANCE
        
    def save_color_scheme(vis):
        filename = input("Enter the filename to save the color scheme: ")
        path = os.path.join("./tmp/cfslam_vis_color/", f"{filename}.json")
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(main.class_colors, f)
        print("Color scheme saved to: ", path)
        
    def load_color_scheme(vis):
        filename = input("Enter the filename to load the color scheme: ")
        path = os.path.join("./tmp/cfslam_vis_color/", f"{filename}.json")
        path = os.path.abspath(path)
        
        if not os.path.exists(path):
            print("File does not exist:", path)
            return
        
        with open(path, "r") as f:
            load_colors = json.load(f)
            
        if len(load_colors) != len(main.class_colors):
            print("Color scheme does not match the class names.")
            return
        main.class_colors = load_colors
        print("Color scheme loaded from: ", path)
    
    def pause_resume(vis):
        main.paused = not main.paused
        
    def toggle_bg(vis):
        main.show_bg = not main.show_bg
        
    def toggle_forward(vis):
        main.forward = not main.forward
        
    def toggle_top(vis):
        main.show_top = not main.show_top
        
    def toggle_cam(vis):
        main.show_cam = not main.show_cam

    def toggle_bbox(vis):
        main.show_bbox = not main.show_bbox
        
    def save_vis_capture(vis):
        vis.poll_events()
        vis.update_renderer()
        
        rendered_image = vis.capture_screen_float_buffer(False)
        rendered_image = np.asarray(rendered_image)
        
        rendered_image = (rendered_image * 255).astype(np.uint8)
        
        filename = input("Enter the filename to save the visualization: ")
        path = os.path.join("./tmp/cfslam_vis/", f"{filename}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.imwrite(path, rendered_image)
        
    def save_camera_params(vis):
        filename = input("Enter the filename to save the camera parameters: ")
        path = os.path.join("./tmp/cfslam_vis_cam/", f"{filename}.json")
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        vis_ctrl = vis.get_view_control()
        params = vis_ctrl.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(path, params)
        print("Camera parameters saved to: ", path)

    def load_camera_params(vis):
        filename = input("Enter the filename to load the camera parameters: ")
        path = os.path.join("./tmp/cfslam_vis_cam/", f"{filename}.json")
        path = os.path.abspath(path)
        
        if not os.path.exists(path):
            print("File does not exist:", path)
            return
        
        vis_ctrl = vis.get_view_control()
        params = o3d.io.read_pinhole_camera_parameters(path)
        vis_ctrl.convert_from_pinhole_camera_parameters(params)
    
    def quit(vis):
        vis.destroy_window()
        print(main.frame_idx)
        exit()
        
    def go_to_frame_idx(vis):
        idx = int(input("Enter the frame index to go to: "))
        if not (0 <= main.frame_idx < len(frame_paths)):
            print("Invalid frame index: ", idx)
            return
        main.frame_idx = idx
        print("Frame index set to: ", main.frame_idx)

    def go_forward_one_frame(vis):
        main.frame_idx += 1
        main.frame_idx = min(main.frame_idx, len(frame_paths)-1)
        
    def go_backward_one_frame(vis):
        main.frame_idx -= 1
        main.frame_idx = max(main.frame_idx, 0)
    
    vis.register_key_callback(ord("R"), color_mode_rgb)
    vis.register_key_callback(ord("C"), color_mode_class)
    vis.register_key_callback(ord("I"), color_mode_inst)
    
    vis.register_key_callback(ord("O"), save_color_scheme)
    vis.register_key_callback(ord("P"), load_color_scheme)
    
    vis.register_key_callback(ord(" "), pause_resume)
    vis.register_key_callback(ord("B"), toggle_bg)
    vis.register_key_callback(ord("F"), toggle_forward)
    vis.register_key_callback(ord("K"), toggle_cam)
    vis.register_key_callback(ord("X"), toggle_bbox)
    vis.register_key_callback(ord("N"), go_to_frame_idx)
    
    vis.register_key_callback(ord("A"), go_backward_one_frame)
    vis.register_key_callback(ord("D"), go_forward_one_frame)

    vis.register_key_callback(ord("T"), toggle_top)
    vis.register_key_callback(ord("S"), save_vis_capture)
    vis.register_key_callback(ord("W"), save_camera_params)
    vis.register_key_callback(ord("E"), load_camera_params)
    vis.register_key_callback(ord("Q"), quit)
    
    t = 0 # How many frames have been rendered
    while True:
        tic = time.time()
        camera_pose, objects, bg_objects = load_frame(frame_paths[main.frame_idx])

        # Hide background objects by setting it to None
        if not main.show_bg:
            bg_objects = None
            
        if bg_objects is not None and not main.show_top:
            # Hide the ceiling
            bg_classes = bg_objects.get_most_common_class()
            bg_objects = bg_objects.slice_by_mask(
                np.asarray(bg_classes) != class_names.index("ceiling")
            )
        
        # Give proper colors to the objects
        if main.color_mode == COLOR_MODE.RGB:
            pass # The default color is RGB, do nothing
        elif main.color_mode == COLOR_MODE.CLASS:
            objects.color_by_most_common_classes(main.class_colors, color_bbox=True)
            if bg_objects is not None:
                bg_objects.color_by_most_common_classes(main.class_colors, color_bbox=True)
        elif main.color_mode == COLOR_MODE.INSTANCE:
            objects.color_by_instance()
            if bg_objects is not None:
                bg_objects.color_by_instance()
                
        # Add geometries from the current frame
        vis.clear_geometries()
        for pcd in objects.get_values("pcd"):
            vis.add_geometry(pcd, reset_bounding_box = t<=3)
            
        if main.show_bbox:
            # for bbox in objects.get_values("bbox"):
            #     vis.add_geometry(bbox, reset_bounding_box = t<=3)
            for obj in objects:
                pcd = obj['pcd']
                bbox = pcd.get_axis_aligned_bounding_box()
                # bbox.color = obj['bbox'].color
                bbox.color = (0, 1, 0)
                vis.add_geometry(bbox, reset_bounding_box = t<=3)
        
        if bg_objects is not None:
            for pcd in bg_objects.get_values("pcd"):
                vis.add_geometry(pcd, reset_bounding_box = t<=3)

        # Set the camera pose
        if args.follow_cam:
            # Set the rendering camera pose to the observation camera pose
            camera_param.extrinsic = np.linalg.inv(to_numpy(camera_pose))
            view_ctrl.convert_from_pinhole_camera_parameters(camera_param)
            view_ctrl.camera_local_translate(forward=-0.4, right=0.0, up=0.0)
        else:
            if main.show_cam and camera_pose is not None:
                camera_frustum = better_camera_frustum(
                    camera_pose, 680//2, 1200//2, scale=0.2, color=[0, 1., 0]
                )
                vis.add_geometry(camera_frustum, reset_bounding_box = t<=3)
        
        vis.poll_events()
        vis.update_renderer()
        
        t += 1
        
        if main.paused:
            continue
        else:
            if main.forward:
                main.frame_idx += 1
                main.frame_idx = min(main.frame_idx, len(frame_paths)-1)
            else:
                main.frame_idx -= 1
                main.frame_idx = max(main.frame_idx, 0)
                
            toc = time.time()
            # print(toc - tic)
            if toc - tic < args.sleep_time:
                time.sleep(args.sleep_time - (toc-tic))
                
    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)