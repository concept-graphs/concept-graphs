# Some utility functions for AI2Thor environment
import copy
from typing import Tuple
from PIL import Image

from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

import prior
from ai2thor.controller import Controller
import torch
from tqdm import trange

def get_top_down_frame(controller):
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["position"]["y"] += 1.1 * max_bound
    pose["farClippingPlane"] = 50
    # pose["fieldOfView"] = 50
    # pose["orthographic"] = False
    # del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = event.third_party_camera_frames[-1]

    # Get a grid visualization of reachable positions
    event = controller.step(action="GetReachablePositions", raise_for_failure=True)
    reachable_positions = event.metadata["actionReturn"]
    xs = [rp["x"] for rp in reachable_positions]
    zs = [rp["z"] for rp in reachable_positions]

    fig, ax = plt.subplots(1, 1)
    ax.scatter(xs, zs)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.set_title("Reachable Positions in the Scene")
    ax.set_aspect("equal")
    # Convert the plot to an image
    fig.canvas.draw()
    top_down_grid = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    top_down_grid = top_down_grid.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return Image.fromarray(top_down_frame), Image.fromarray(top_down_grid)

def adjust_ai2thor_pose(pose):
    '''
    Adjust the camera pose from the one used in Unity to that in Open3D.
    '''
    # Transformation matrix to flip Y-axis
    flip_y = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Separate rotation and translation
    rotation = pose[:3, :3]
    translation = pose[:3, 3]

    # Adjust rotation and translation separately
    adjusted_rotation = flip_y[:3, :3] @ rotation @ flip_y[:3, :3]
    adjusted_translation = flip_y[:3, :3] @ translation

    # Reconstruct the adjusted camera pose
    adjusted_pose = np.eye(4)
    adjusted_pose[:3, :3] = adjusted_rotation
    adjusted_pose[:3, 3] = adjusted_translation
    
    R = Rotation.from_euler('x', 180, degrees=True).as_matrix()
    R_homogeneous = np.eye(4)
    R_homogeneous[:3, :3] = R
    
    T_open3d_rotated = R_homogeneous @ adjusted_pose
    
    adjusted_pose = T_open3d_rotated

    return adjusted_pose

def adjust_ai2thor_pose_batch(poses):
    '''
    Adjust the camera poses from the one used in Unity to that in Open3D.
    '''
    N = poses.shape[0]
    
    # Transformation matrix to flip Y-axis
    flip_y = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    flip_y = np.repeat(flip_y[None, :, :], N, axis=0) # shape (N, 4, 4)

    # Separate rotation and translation
    rotation = poses[:, :3, :3] # shape (N, 3, 3)
    translation = poses[:, :3, 3] # shape (N, 3)

    # Adjust rotation and translation separately
    adjusted_rotation = np.einsum('nij,njk,nkl->nil', flip_y[:, :3, :3], rotation, flip_y[:, :3, :3]) # shape (N, 3, 3)
    adjusted_translation = np.einsum('nij,nj->ni', flip_y[:, :3, :3], translation) # shape (N, 3)

    # Reconstruct the adjusted camera pose
    adjusted_pose = np.eye(4).reshape(1, 4, 4).repeat(N, axis=0) # shape (N, 4, 4)
    adjusted_pose[:, :3, :3] = adjusted_rotation
    adjusted_pose[:, :3, 3] = adjusted_translation

    # Rotation by 180 degrees around x-axis
    R = Rotation.from_euler('x', 180, degrees=True).as_matrix()
    R_homogeneous = np.eye(4)
    R_homogeneous[:3, :3] = R

    R_homogeneous = np.repeat(R_homogeneous[None, :, :], N, axis=0) # shape (N, 4, 4)
    
    adjusted_pose = np.einsum('nij,njk->nik', R_homogeneous, adjusted_pose) # shape (N, 4, 4)

    return adjusted_pose

def adjust_ai2thor_batch_torch(poses):
    '''
    Adjust the camera poses from the one used in Unity to that in Open3D.
    
    Args:
        poses: torch.Tensor, shape (N, 4, 4)
        
    Returns:
        adjusted_pose: torch.Tensor, shape (N, 4, 4)
    '''
    N = poses.shape[0]
    
    # Transformation matrix to flip Y-axis
    flip_y = torch.tensor([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).to(poses.device).type(poses.dtype)

    flip_y = flip_y[None, :, :].expand(N, -1, -1) # shape (N, 4, 4)

    # Separate rotation and translation
    rotation = poses[:, :3, :3] # shape (N, 3, 3)
    translation = poses[:, :3, 3] # shape (N, 3)

    # Adjust rotation and translation separately
    adjusted_rotation = flip_y[:, :3, :3].bmm(rotation).bmm(flip_y[:, :3, :3]) # shape (N, 3, 3)
    adjusted_translation = flip_y[:, :3, :3].bmm(translation.unsqueeze(-1)).squeeze(-1) # shape (N, 3)

    # Reconstruct the adjusted camera pose
    adjusted_pose = torch.eye(4).to(poses.device).type(poses.dtype).unsqueeze(0).expand(N, -1, -1) # shape (N, 4, 4)
    adjusted_pose = adjusted_pose.clone()
    adjusted_pose[:, :3, :3] = adjusted_rotation
    adjusted_pose[:, :3, 3] = adjusted_translation

    # Rotation by 180 degrees around x-axis
    R = Rotation.from_euler('x', 180, degrees=True).as_matrix()
    R_homogeneous = np.eye(4)
    R_homogeneous[:3, :3] = R

    R_homogeneous = torch.from_numpy(R_homogeneous).type(poses.dtype).to(poses.device)
    R_homogeneous = R_homogeneous[None, :, :].expand(N, -1, -1) # shape (N, 4, 4)
    
    adjusted_pose = R_homogeneous.bmm(adjusted_pose) # shape (N, 4, 4)

    return adjusted_pose

def depth2xyz(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    '''
    Convert depth image to 3D XYZ image in the camera coordinate frame

    Args:
        depth: depth image, shape (H, W), in meters
        K: camera intrinsics matrix, shape (3, 3)
        
    Returns:
        xyz_camera: 3D XYZ image in the camera coordinate frame, shape (H, W, 3)
    '''
    frame_size = depth.shape[:2]
    
    x = np.arange(0, frame_size[1]) 
    y = np.arange(frame_size[0], 0, -1) 
    xx, yy = np.meshgrid(x, y)

    xx = xx.flatten()
    yy = yy.flatten()
    zz = depth.flatten()
    xyz = np.stack([xx, yy, zz], axis=1)
    xyz = xyz.astype(np.float32)

    # Compute the XYZ map in camera coordinates
    x_camera = (xx - K[0, 2]) * zz / K[0, 0]
    y_camera = (yy - K[1, 2]) * zz / K[1, 1]
    z_camera = zz
    xyz_camera = np.stack([x_camera, y_camera, z_camera], axis=1)
    
    xyz_camera = xyz_camera.reshape((*frame_size, 3))
    return xyz_camera

def transform_xyz(xyz: np.ndarray, pose: np.ndarray) -> np.ndarray:
    '''
    Transform the 3D XYZ image using the input pose matrix
    
    Args:
        xyz: 3D XYZ image, shape (H, W, 3)
        pose: 4x4 pose matrix, shape (4, 4)
         
    Returns:
        xyz_transformed: transformed 3D XYZ image, shape (H, W, 3)
    '''
    xyz_flatten = xyz.reshape(-1, 3)
    xyz_transformed = pose @ np.concatenate([xyz_flatten, np.ones((xyz_flatten.shape[0], 1))], axis=1).T
    xyz_transformed = xyz_transformed.T[:, :3]
    xyz_transformed = xyz_transformed.reshape(xyz.shape)
    return xyz_transformed

def get_scene(scene_name):
    # By default, use scene from AI2THOR
    # If the scene name starts with train, val, or test, use the scene from ProcTHOR
    scene = scene_name
    if (
        scene_name.startswith("train")
        or scene_name.startswith("val")
        or scene_name.startswith("test")
    ):
        dataset = prior.load_dataset("procthor-10k")
        scene = dataset[scene_name.split("_")[0]][int(scene_name.split("_")[1])]
    return scene

def compute_intrinsics(vfov, height, width):
    """
    Compute the camera intrinsics matrix K from the
    vertical field of view (in degree), height, and width.
    """
    # For Unity, the field view is the vertical field of view.
    f = height / (2 * np.tan(np.deg2rad(vfov) / 2))
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])


def compute_pose(position: dict, rotation: dict) -> np.ndarray:
    """
    Compute the camera extrinsics matrix from the position and rotation.

    Note that in Unity, XYZ follows the left-hand rule, with Y pointing up.
    See: https://docs.unity3d.com/560/Documentation/Manual/Transforms.html
    In the camera coordinate, Z is the viewing direction, X is right, and Y is up. 
    See: https://library.vuforia.com/device-tracking/spatial-frame-reference
    Euler angles are in degrees and in Rotation is done in the ZXY order.
    See: https://docs.unity3d.com/ScriptReference/Transform-eulerAngles.html
    """
    x, y, z = position["x"], position["y"], position["z"]
    rx, ry, rz = rotation["x"], rotation["y"], rotation["z"]

    # Compute the Rotation matrix
    R = Rotation.from_euler("zxy", [rz, rx, ry], degrees=True).as_matrix()

    # Compute the translation matrix
    t = np.array([x, y, z])

    # Represent the pose in homogeneous coordinates
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def compute_posrot(T: np.ndarray) -> Tuple[dict, dict]:
    """
    Decompose the camera extrinsics matrix into position and rotation.

    This function reverses the operation performed in `compute_pose`, considering
    Unity's conventions, left-hand coordinate system, and ZXY Euler angle rotation order.
    """

    # Extract the rotation matrix from the first 3x3 elements
    R = T[:3, :3]

    # Extract the translation vector from the last column
    t = T[:3, 3]

    # Compute the Euler angles from the rotation matrix
    euler_angles = Rotation.from_matrix(R).as_euler("zxy", degrees=True)

    # Convert the position and rotation into dictionaries
    position = {"x": t[0], "y": t[1], "z": t[2]}
    rotation = {"x": euler_angles[1], "y": euler_angles[2], "z": euler_angles[0]}

    return position, rotation

def get_agent_pose_from_event(event) -> np.ndarray:
    '''
    Compute the 4x4 agent pose matrix from the event
    '''
    position = event.metadata["agent"]["position"]
    rotation = event.metadata["agent"]["rotation"]

    # Compute the agent pose (position and rotation of agent's body in global 3D space)
    agent_pose = compute_pose(position, rotation)
    return agent_pose

def get_camera_pose_from_event(event) -> np.ndarray:
    '''
    Compute the 4x4 camera pose matrix from the event
    This is different from the agent pose!
    '''
    camera_position = event.metadata['cameraPosition']
    camera_rotation = copy.deepcopy(event.metadata["agent"]["rotation"])
    camera_rotation['x'] = event.metadata['agent']['cameraHorizon']
    camera_pose = compute_pose(camera_position, camera_rotation)
    return camera_pose


def sample_pose_random(controller: Controller, n_poses: int):
    reachable_positions = controller.step(action="GetReachablePositions").metadata[
        "actionReturn"
    ]

    # Convert the positions to numpy array
    reachable_np = np.array([[p["x"], p["y"], p["z"]] for p in reachable_positions])
    print(reachable_np)

    # Generate a list of poses for taking pictures
    sampled_poses = []
    for i in trange(n_poses):
        # randomly sample a position
        position = np.random.choice(reachable_positions)

        # randomly sample a rotation
        rot_y = np.random.uniform(-180, 180)
        rotation = dict(x=0, y=rot_y, z=0)

        sampled_poses.append(
            dict(
                position=position,
                rotation=rotation,
                horizon=0,
                standing=True,
            )
        )

    return sampled_poses


def sample_pose_uniform(controller: Controller, n_positions: int):
    """
    Uniformly sample n_positions from the reachable positions
    for each position, uniformly sample 8 rotations (0, 45, 90, 135, 180, 225, 270, 315)
    """
    reachable_positions = controller.step(action="GetReachablePositions").metadata[
        "actionReturn"
    ]

    # Convert the positions to numpy array
    reachable_np = np.array([[p["x"], p["y"], p["z"]] for p in reachable_positions])
    # Sort the positions by x, z
    sort_idx = np.lexsort((reachable_np[:, 2], reachable_np[:, 0]))
    reachable_positions = [reachable_positions[i] for i in sort_idx]

    # Randomly sample n_positions. This is a temporal hack for uniform sampling.
    if n_positions < 0:
        n_positions = len(reachable_positions)
    else:
        n_positions = min(n_positions, len(reachable_positions))

    sampled_positions = np.random.choice(
        reachable_positions, n_positions, replace=False
    )

    # Generate a list of poses for taking pictures
    sampled_poses = []
    for position in sampled_positions:
        for rot_y in [0, 45, 90, 135, 180, 225, 270, 315]:
            rotation = dict(x=0, y=rot_y, z=0)

            sampled_poses.append(
                dict(
                    position=position,
                    rotation=rotation,
                    horizon=0,
                    standing=True,
                )
            )

    return sampled_poses