"""
Preprocess an unzipped .r3d file to the Record3DDataset format.
"""

import glob
import json
import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import liblzfse  # https://pypi.org/project/pyliblzfse/
import numpy as np
import png  # pip install pypng
import torch
import tyro
from natsort import natsorted
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm, trange

@dataclass
class ProgramArgs:
    datapath: str = "/home/krishna/data/record3d/krishna-bcs-room"


def load_depth(filepath):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

    # depth_img = depth_img.reshape((640, 480))  # For a FaceID camera 3D Video
    depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video

    return depth_img


def load_conf(filepath):
    with open(filepath, 'rb') as conf_fh:
        raw_bytes = conf_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        conf_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)

    # depth_img = depth_img.reshape((640, 480))  # For a FaceID camera 3D Video
    conf_img = conf_img.reshape((256, 192))  # For a LiDAR 3D Video

    return conf_img


def load_color(filepath):
    img = cv2.imread(filepath)
    return cv2.resize(img, (192, 256))


def write_color(outpath, img):
    cv2.imwrite(outpath, img)


def write_depth(outpath, depth):
    depth = depth * 1000
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(outpath)

def write_conf(outpath, conf):
    np.save(outpath, conf)

def write_pose(outpath, pose):
    np.save(outpath, pose.astype(np.float32))


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
    
    # Keys in metadata dict
    # h, w, K, fps, dw, dh, initPose, poses, cameraType, frameTimestamps
    # print(metadata.keys())

    poses = get_poses(metadata)
    intrinsics_dict = get_intrinsics(metadata)
    
    color_paths = natsorted(glob.glob(os.path.join(args.datapath, "rgbd", "*.jpg")))
    depth_paths = natsorted(glob.glob(os.path.join(args.datapath, "rgbd", "*.depth")))
    conf_paths = natsorted(glob.glob(os.path.join(args.datapath, "rgbd", "*.conf")))

    os.makedirs(os.path.join(args.datapath, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(args.datapath, "conf"), exist_ok=True)
    os.makedirs(os.path.join(args.datapath, "depth"), exist_ok=True)
    os.makedirs(os.path.join(args.datapath, "poses"), exist_ok=True)

    cfg = {}
    cfg["dataset_name"] = "record3d"
    cfg["camera_params"] = {}
    cfg["camera_params"]["image_height"] = intrinsics_dict["h"]
    cfg["camera_params"]["image_width"] = intrinsics_dict["w"]
    cfg["camera_params"]["fx"] = intrinsics_dict["fx"].item()
    cfg["camera_params"]["fy"] = intrinsics_dict["fy"].item()
    cfg["camera_params"]["cx"] = intrinsics_dict["cx"].item()
    cfg["camera_params"]["cy"] = intrinsics_dict["cy"].item()
    cfg["camera_params"]["png_depth_scale"] = 1000.0
    print(cfg)
    with open(os.path.join(args.datapath, "dataconfig.yaml"), "w") as f:
        yaml.dump(cfg, f)

    for i in trange(len(color_paths)):
        color = load_color(color_paths[i])
        depth = load_depth(depth_paths[i])
        conf = load_conf(conf_paths[i])
        basename = os.path.splitext(os.path.basename(color_paths[i]))[0]
        # color_path = os.path.splitext(os.path.basename(color_paths[i]))[0] + ".png"
        write_color(os.path.join(args.datapath, "rgb", basename + ".png"), color)
        # depth_path = os.path.splitext(os.path.basename(depth_paths[i]))[0] + ".png"
        write_depth(os.path.join(args.datapath, "depth", basename + ".png"), depth)
        # conf_path = os.path.splitext(os.path.basename(conf_paths[i]))[0] + ".npy"
        write_conf(os.path.join(args.datapath, "conf", basename + ".npy"), conf)
        write_pose(os.path.join(args.datapath, "poses", basename + ".npy"), poses[i])
        # c2w = poses[i]
        # frame = {
        #     "file_path": os.path.join("rgb", color_path),
        #     "depth_path": os.path.join("depth", depth_path),
        #     "conf_path": os.path.join("conf", conf_path),
        #     "transform_matrix": c2w.tolist(),
        # }
        # frames.append(frame)


if __name__ == "__main__":
    main()
