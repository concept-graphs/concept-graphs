import copy
import gc
import os
import pickle
import gzip
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rich

import open3d as o3d
from PIL import Image
import torch
import torch.nn.functional as F
import open_clip
from gradslam.structures.pointclouds import Pointclouds

from conceptgraph.slam.slam_classes import MapObjectList

from conceptgraph.llava.llava_model import LLaVaChat
from conceptgraph.utils.image import crop_image_pil
try: LLAVA_CKPT_PATH = os.environ["LLAVA_CKPT_PATH"]
except KeyError: raise ValueError("Please set the environment variable LLAVA_CKPT_PATH to the path of the LLaVA checkpoint")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--rgb_pcd_path", type=str, default=None)
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # rich console for pretty printing
    # console = rich.console.Console()

    result_path = args.result_path
    rgb_pcd_path = args.rgb_pcd_path

    if rgb_pcd_path is not None:        
        pointclouds = Pointclouds.load_pointcloud_from_h5(rgb_pcd_path)
        global_pcd = pointclouds.open3d(0, include_colors=True)

    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)
        
    class_names = results['class_names']
    class_colors = results['class_colors']
        
    objects = MapObjectList()
    objects.load_serializable(results['objects'])
    
    cmap = matplotlib.colormaps.get_cmap("turbo")
    
    pcds = copy.deepcopy(objects.get_values("pcd"))
    bboxes = copy.deepcopy(objects.get_values("bbox"))
    
    # Get the color for each object when colored by their class
    object_classes = []
    for i in range(len(objects)):
        obj = objects[i]
        pcd = pcds[i]
        obj_classes = np.asarray(obj['class_id'])
        # Get the most common class for this object as the class
        values, counts = np.unique(obj_classes, return_counts=True)
        obj_class = values[np.argmax(counts)]
        object_classes.append(obj_class)
        
    # Color the object by their classes
    for i in range(len(objects)):
        pcd = pcds[i]
        obj_class = object_classes[i]
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(
                class_colors[str(obj_class)],
                (len(pcd.points), 1)
            )
        )
        
    # Build a global point cloud and record the object indices
    global_pcd = o3d.geometry.PointCloud()
    global_pcd.points = o3d.utility.Vector3dVector(np.concatenate([pcd.points for pcd in pcds]))
    global_pcd.colors = o3d.utility.Vector3dVector(np.concatenate([pcd.colors for pcd in pcds]))
    object_indices = np.concatenate([np.full(len(pcd.points), i) for i, pcd in enumerate(pcds)])
    
    # chat = LLaVaChat(LLAVA_CKPT_PATH, "multimodal", 1)
    chat = LLaVaChat(LLAVA_CKPT_PATH, "default", 1)
    print("LLaVA chat initialized...")
    
    while True:
        # Set the title of the window
        vis = o3d.visualization.VisualizerWithEditing()

        if result_path is not None:
            vis.create_window(window_name=f'Open3D - {os.path.basename(result_path)}', width=1280, height=720)
        else:
            vis.create_window(window_name=f'Open3D', width=1280, height=720)

        vis.add_geometry(global_pcd)
            
        print("")
        print("==> Please points using [shift + left click]")
        print("   Press [shift + right click] to undo point picking")
        print("   Only the last point picking will be considered")
        print("==> Afther picking points, press q for close the window")
        
        # Render the scene
        vis.run()
        
        vis.destroy_window()
        
        picked_indices = vis.get_picked_points()
        if len(picked_indices) == 0:
            print("No points picked, exiting...")
            exit()
            
        chat.reset()
            
        picked_obj_idx = object_indices[picked_indices[-1]]
        picked_obj = objects[picked_obj_idx]
        print(f"Object index: {picked_obj_idx}")

        picked_obj_classes = np.asarray(picked_obj['class_id'])
        values, counts = np.unique(picked_obj_classes, return_counts=True)
        print("Object class:", end=" ")
        for i in counts.argsort()[::-1]:
            print(f"{class_names[values[i]]}({counts[i]})", end=" ")
        print("")
        picked_obj_class_name = class_names[values[np.argmax(counts)]]
        
        conf = picked_obj['conf']
        conf = picked_obj['n_points']
        conf = np.asarray(conf)
        idx_sort = np.argsort(conf)[::-1]
        
        object_crops = []
        crop_features = []
        for i, idx_det in enumerate(idx_sort[:5]):
            # wrong_path = picked_obj['color_path'][idx_det]
            # correct_prefix = '/home/sacha/data/scan12/'
            # wrong_prefix = '/home/kuwajerw/local_data/azure/liam_lab_w_objects/scan12/'
            # correct_path = wrong_path.replace(wrong_prefix, correct_prefix)

            # image = Image.open(correct_path)

            image = Image.open(picked_obj['color_path'][idx_det])
            mask = picked_obj['mask'][idx_det]
            class_id = picked_obj['class_id'][idx_det]
            xyxy = picked_obj['xyxy'][idx_det]
            class_name = class_names[class_id]
            
            x1, y1, x2, y2 = xyxy
            image_crop = crop_image_pil(image, x1, y1, x2, y2, 10)

            image_tensor = chat.image_processor.preprocess(image_crop, return_tensors="pt")[
                "pixel_values"
            ][0]
            image_feature = chat.encode_image(image_tensor[None, ...].half().cuda())
            
            object_crops.append(image_crop)
            crop_features.append(image_feature)
            
        # Show the crops using plt
        fig, axes = plt.subplots(1, len(object_crops), figsize=(10, 3))
        if len(object_crops) == 1:
            axes = [axes]
        for i, image_crop in enumerate(object_crops):
            axes[i].imshow(image_crop)
            axes[i].set_title(f"{picked_obj_class_name} {i}")
            axes[i].axis("off")
        fig.suptitle("Image crops used for LLaVA")
        plt.show(block=False)
        
        # print("Using the LLaVA feature from the most confident detection...")
        # llava_feat = crop_features[0]
        
        print(f"Using the mean feature from {len(crop_features)} image crops...")
        print(crop_features[0].shape)
        llava_feat = torch.stack(crop_features, dim=0).mean(dim=0)
        print(llava_feat.shape)
        
        query = "What is the central object in this image?"
        outputs = chat(query=query, image_features=llava_feat.half().cuda())
        # console.print("[bold red]User:[/bold red] " + query)
        print("User: " + query)
        # print(outputs)
        print("LLaVA: " + outputs)

        while True:
            query = input("Enter a query ('q' to quit): ")
            if query == "q":
                break
            # console.print("[bold red]User:[/bold red] " + query)
            print("User: " + query)
            outputs = chat(query=query, image_features=None)
            # print(outputs)
            print("LLaVA: " + outputs)

            
        plt.close(fig)
        
        
            
            
        
            
            