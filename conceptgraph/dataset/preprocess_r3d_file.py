"""
Preprocess an unzipped .r3d file to the Record3DDataset format with adjusted image resolution.
"""

import glob
import json
import os
import cv2
import liblzfse
import numpy as np
import png
import tyro
import yaml
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm, trange
from typing import List, Tuple, Union
from natsort import natsorted
from pathlib import Path

@dataclass
class ProgramArgs:
    datapath = "/home/kuwajerw/new_local_data/new_record3d/ali_apartment/co_store"
    output_dir = None  # Optional, set dynamically if not provided

desired_width = 1440
desired_height = 1920

def load_depth(filepath):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)
        depth_img = depth_img.reshape((256, 192))  # Original resolution
        depth_img = resize_depth(depth_img, desired_width, desired_height)
    return depth_img

def load_conf(filepath):
    with open(filepath, 'rb') as conf_fh:
        raw_bytes = conf_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        conf_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)
        conf_img = conf_img.reshape((256, 192))  # Original resolution
        conf_img = resize_depth(conf_img, desired_width, desired_height)  # Using the same resizing function as depth
    return conf_img

def load_color(filepath):
    img = cv2.imread(filepath)
    resized_img = cv2.resize(img, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)
    return resized_img

def resize_depth(depth_img, desired_width, desired_height):
    return cv2.resize(depth_img, (desired_width, desired_height), interpolation=cv2.INTER_NEAREST)

def write_color(outpath, img):
    cv2.imwrite(outpath, img)

def write_depth(outpath, depth):
    depth = depth * 1000
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(outpath)

def write_conf(outpath, conf):
    np.save(outpath, conf)

def write_conf_img(outpath, conf):
    conf_img = Image.fromarray(conf)
    conf_img.save(outpath)

def write_pose(outpath, pose):
    np.save(outpath, pose.astype(np.float32))

def adjust_intrinsics(intrinsics_dict, original_width, original_height):
    width_scale = desired_width / original_width
    height_scale = desired_height / original_height
    
    intrinsics_dict["fx"] *= width_scale
    intrinsics_dict["fy"] *= height_scale
    intrinsics_dict["cx"] *= width_scale
    intrinsics_dict["cy"] *= height_scale
    intrinsics_dict["w"] = desired_width
    intrinsics_dict["h"] = desired_height
    
    return intrinsics_dict

# The rest of your functions like get_poses and get_intrinsics remain unchanged

def get_poses(metadata_dict: dict) -> int:
    """Converts Record3D's metadata dict into pose matrices needed by nerfstudio
    Args:
        metadata_dict: Dict containing Record3D metadata
    Returns:
        np.array of pose matrices for each image of shape: (num_images, 4, 4)
    """

    poses_data = np.array(metadata_dict["poses"])  # (N, 3, 4)
    # NB: Record3D / scipy use "scalar-last" format quaternions (x y z w)
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    camera_to_worlds = np.concatenate(
        [Rotation.from_quat(poses_data[:, :4]).as_matrix(), poses_data[:, 4:, None]],
        axis=-1,
    ).astype(np.float32)

    homogeneous_coord = np.zeros_like(camera_to_worlds[..., :1, :])
    homogeneous_coord[..., :, 3] = 1
    camera_to_worlds = np.concatenate([camera_to_worlds, homogeneous_coord], -2)
    
    return camera_to_worlds


def get_intrinsics(metadata_dict: dict, downscale_factor: float = 7.5) -> int:
    """Converts Record3D metadata dict into intrinsic info needed by nerfstudio
    Args:
        metadata_dict: Dict containing Record3D metadata
        downscale_factor: factor to scale RGB image by (usually scale factor is
            set to 7.5 for record3d 1.8 or higher -- this is the factor that downscales
            RGB images to lidar)
    Returns:
        dict with camera intrinsics keys needed by nerfstudio
    """

    # Camera intrinsics
    K = np.array(metadata_dict["K"]).reshape((3, 3)).T
    K = K / downscale_factor
    K[2, 2] = 1.0
    focal_length = K[0, 0]

    H = metadata_dict["h"]
    W = metadata_dict["w"]

    H = int(H / downscale_factor)
    W = int(W / downscale_factor)

    # # TODO(akristoffersen): The metadata dict comes with principle points,
    # # but caused errors in image coord indexing. Should update once that is fixed.
    # cx, cy = W / 2, H / 2
    cx, cy = K[0, 2], K[1, 2]

    intrinsics_dict = {
        "fx": focal_length,
        "fy": focal_length,
        "cx": cx,
        "cy": cy,
        "w": W,
        "h": H,
    }

    return intrinsics_dict

def main():
    args = tyro.cli(ProgramArgs)
    
    metadata = None
    with open(os.path.join(args.datapath, "metadata"), "r") as f:
        metadata = json.load(f)
        
    # If output_dir is not specified, set it to a "preprocessed" folder inside datapath parent folder
    if args.output_dir is None:
        datapath = Path(args.datapath)
        args.output_dir = str(datapath.parent / (datapath.name + "_preprocessed"))
    
    print(f"Preprocessing Record3D data from \n{args.datapath} to \n{args.output_dir}")

    original_width, original_height = 192, 256 # Original resolution for scaling intrinsics
    poses = get_poses(metadata)
    intrinsics_dict = get_intrinsics(metadata)
    intrinsics_dict = adjust_intrinsics(intrinsics_dict, original_width, original_height)
    
    color_paths = natsorted(glob.glob(os.path.join(args.datapath, "rgbd", "*.jpg")))
    depth_paths = natsorted(glob.glob(os.path.join(args.datapath, "rgbd", "*.depth")))
    conf_paths = natsorted(glob.glob(os.path.join(args.datapath, "rgbd", "*.conf")))

    # Modify paths in os.makedirs to use args.output_dir
    os.makedirs(os.path.join(args.output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "conf"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "poses"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "conf_images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "high_conf_depth"), exist_ok=True)




    cfg = {
        "dataset_name": "record3d",
        "camera_params": {
            "image_height": intrinsics_dict["h"],
            "image_width": intrinsics_dict["w"],
            "fx": intrinsics_dict["fx"].item(),
            "fy": intrinsics_dict["fy"].item(),
            "cx": intrinsics_dict["cx"].item(),
            "cy": intrinsics_dict["cy"].item(),
            "png_depth_scale": 1000.0
            }
        }

    with open(os.path.join(args.output_dir, "dataconfig.yaml"), "w") as f:
        yaml.dump(cfg, f)

    for i in trange(len(color_paths)):
        color = load_color(color_paths[i])
        depth = load_depth(depth_paths[i])
        conf = load_conf(conf_paths[i])
        
        # New: Generate high confidence depth image
        high_conf_mask = conf == 2  # Create a mask where conf is 2
        high_conf_depth = np.where(high_conf_mask, depth, 0)  # Set depth to 0 where conf is not 2

        basename = os.path.splitext(os.path.basename(color_paths[i]))[0]
        # color_path = os.path.splitext(os.path.basename(color_paths[i]))[0] + ".png"
        write_color(os.path.join(args.output_dir, "rgb", basename + ".png"), color)
        # depth_path = os.path.splitext(os.path.basename(depth_paths[i]))[0] + ".png"
        write_depth(os.path.join(args.output_dir, "depth", basename + ".png"), depth)
        # conf_path = os.path.splitext(os.path.basename(conf_paths[i]))[0] + ".npy"
        write_conf(os.path.join(args.output_dir, "conf", basename + ".npy"), conf)
        write_conf_img(os.path.join(args.output_dir, "conf_images", basename + ".png"), conf)
        write_pose(os.path.join(args.output_dir, "poses", basename + ".npy"), poses[i])
        write_depth(os.path.join(args.output_dir, "high_conf_depth", basename + ".png"), high_conf_depth)

if __name__ == "__main__":
    main()
