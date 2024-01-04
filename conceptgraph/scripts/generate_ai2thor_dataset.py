import copy
import json
import os
from pathlib import Path
import random
import pickle
import warnings

from PIL import Image
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from tqdm import trange
import argparse

from conceptgraph.utils.ai2thor import (
    get_agent_pose_from_event, 
    get_camera_pose_from_event, 
    get_top_down_frame,
    compute_intrinsics, 
    compute_pose, 
    get_scene, 
    sample_pose_uniform, 
    sample_pose_random
)
from conceptgraph.ai2thor.rearrange import rearrange_objects

NOT_TO_REMOVE = [
    "Wall",
    "Floor",
    "Window",
    "Doorway",
    "Room",
]

def generate_obs_from_poses(
    controller,
    K,
    sampled_poses,
    save_root,
    depth_scale=1000.0,
):
    color_path_temp = save_root + "/color/{:06d}.png"
    depth_path_temp = save_root + "/depth/{:06d}.png"
    instance_path_temp = save_root + "/instance/{:06d}.png"
    pose_path_temp = save_root + "/pose/{:06d}.txt"

    intrinsics_path = save_root + "/intrinsics.txt"
    obj_meta_path = save_root + "/obj_meta.json"
    color_to_object_id_path = save_root + "/color_to_object_id.pkl"
    object_id_to_color_path = save_root + "/object_id_to_color.pkl"
    video_save_path = save_root + "/rgb_video.mp4"

    # Generate and save images
    frames = []
    for i in trange(len(sampled_poses)):
        pose = sampled_poses[i]
        # Teleport the agent to the position and rotation
        
        # limit the horizon to [-30, 60]
        horizon = pose["horizon"]
        horizon = max(min(horizon, 60-1e-6), -30+1e-6)
        
        event = controller.step(
            action="Teleport",
            position=pose["position"],
            rotation=pose["rotation"],
            horizon=horizon,
            standing=pose["standing"],
            forceAction=True,
        )

        if not event.metadata["lastActionSuccess"]:
            # raise Exception(event.metadata["errorMessage"])

            # Seems that the teleportation failures are based on position. 
            # Once it fails on a position, it will fail on all orientations.
            # Therefore, we can simply skip these failed trials. 
            print("Failed to teleport to the position.", pose["position"], pose["rotation"])
            continue

        color = np.asarray(event.frame).copy()
        depth = np.asarray(event.depth_frame).copy()
        instance = np.asarray(event.instance_segmentation_frame).copy()

        # Compute the agent and camera pose. They are different!
        agent_pose = get_agent_pose_from_event(event)
        camera_pose = get_camera_pose_from_event(event)

        color_path = color_path_temp.format(i)
        depth_path = depth_path_temp.format(i)
        instance_path = instance_path_temp.format(i)
        pose_path = pose_path_temp.format(i)

        os.makedirs(os.path.dirname(color_path), exist_ok=True)
        imageio.imwrite(color_path, color)
        
        if args.save_video:
            frames.append(color)

        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
        # Cut off the depth at 15 meters 
        # some points are outside the house are handled later. 
        depth[depth > 15] = 0
        depth_png = np.round(depth * depth_scale).astype(np.uint16)
        imageio.imwrite(depth_path, depth_png)
        
        os.makedirs(os.path.dirname(instance_path), exist_ok=True)
        imageio.imwrite(instance_path, instance)

        os.makedirs(os.path.dirname(pose_path), exist_ok=True)
        np.savetxt(pose_path, camera_pose)

    np.savetxt(intrinsics_path, K)
    
    # Save the objects information
    # Since we do not change the state, we can simply use the last event.
    # but the `visible` property does vary across frames, and thus they are not indicative in testing. 
    obj_meta = controller.last_event.metadata["objects"]
    with open(obj_meta_path, "w") as f:
        json.dump(obj_meta, f)
    
    # Save the color from/to object id mapping - they are global and constant across all frames/events. 
    # They are  tupled-indexed dict, and thus cannot be saved as JSON files. 
    color_to_object_id = event.color_to_object_id
    with open(color_to_object_id_path, "wb") as f:
        pickle.dump(color_to_object_id, f)
        
    object_id_to_color = event.object_id_to_color
    with open(object_id_to_color_path, "wb") as f:
        pickle.dump(object_id_to_color, f)
        
    if args.save_video:
        imageio.mimsave(video_save_path, frames, fps=20)
        print("Saved video to", video_save_path)

def sample_pose_from_file(traj_file):
    # Load the trajectory file (json)
    with open(traj_file, "r") as f:
        traj = json.load(f)

    sampled_poses = []
    for log in traj["agent_logs"]:
        sampled_poses.append(
            {
                "position": log["position"],
                "rotation": log["rotation"],
                "horizon": log["cameraHorizon"],
                "standing": log["isStanding"],
            }
        )

    return sampled_poses

def is_removeable(obj, level: int):
    if level == 1: # all objects except those in NOT_TO_REMOVE
        return obj['objectType'] not in NOT_TO_REMOVE
    elif level == 2: # objects that are pickupable or moveable
        return obj['pickupable'] or obj['moveable']
    elif level == 3: # objects that are pickupable
        return obj['pickupable']

def randomize_scene(args, controller) -> list[str]|None:
    '''
    Since we want to keep track of which objects are removed, but it is not done in ai2thor
    So we will keep of a list of object ids that are kept in the scene. 
    if no object is removed from the scene, then return None.
    '''
    if args.randomize_lighting:
        controller.step(
            action="RandomizeLighting",
            brightness=(0.5, 1.5),
            randomizeColor=True,
            hue=(0, 1),
            saturation=(0.5, 1),
            synchronized=False,
        )
        
    if args.randomize_material:
        controller.step(
            action="RandomizeMaterials",
            useTrainMaterials=None,
            useValMaterials=None,
            useTestMaterials=None,
            inRoomTypes=None
        )
        
    # Randomly remove objects
    obj_list = controller.last_event.metadata["objects"]
    removed_object_ids = []
    if args.randomize_remove_ratio > 0.0:
        print("Before randomization, there are {} objects in the scene".format(len(obj_list)))
        for obj in obj_list:
            if is_removeable(obj, args.randomize_remove_level) and \
                random.random() < args.randomize_remove_ratio:
                controller.step(
                    action="DisableObject",
                    objectId=obj['objectId'],
                )
                removed_object_ids.append(obj['objectId'])
        print("After randomization, there are {} objects in the scene".format(
            len(controller.last_event.metadata["objects"])
        ))
    
    # Randomly move objects
    starting_poses, target_poses = None, None
    if args.randomize_move_pickupable_ratio > 0.0 or args.randomize_move_moveable_ratio > 0.0:
        starting_poses, target_poses = rearrange_objects(
            controller = controller,
            pickupable_move_ratio = args.randomize_move_pickupable_ratio,
            moveable_move_ratio = args.randomize_move_moveable_ratio,
            reset = False,
        )

    randomization_log = {
        "removed_object_ids": removed_object_ids,
        "starting_poses": starting_poses,
        "target_poses": target_poses,
        "randomize_lighting": args.randomize_lighting,
        "randomize_material": args.randomize_material,
    }

    return randomization_log

def randomize_scene_from_log(controller, randomization_log):
    if randomization_log['randomize_lighting']:
        warnings.warn("randomize_lighting from log file is not implemented yet")
    if randomization_log['randomize_material']:
        warnings.warn("randomize_material from log file is not implemented yet")
        
    # Remove some objects
    removed_object_ids = randomization_log['removed_object_ids']
    if len(removed_object_ids) > 0:
        for obj_id in removed_object_ids:
            event = controller.step(
                action="DisableObject",
                objectId=obj_id,
            )
            if not event.metadata['lastActionSuccess']:
                warnings.warn("Failed to remove object {}".format(obj_id))
                print(event.metadata['errorMessage'])
                
    # Set object poses
    target_poses = randomization_log['target_poses']
    if target_poses is not None:
        event = controller.step(
            action="SetObjectPoses",
            objectPoses=target_poses
        )
        if not event.metadata['lastActionSuccess']:
            warnings.warn("Failed to set object poses")
            print(event.metadata['errorMessage'])

def load_or_randomize_scene(args, controller):
    randomization_file_path = args.save_root + "/randomization.json"
    
    if os.path.exists(randomization_file_path):
        with open(randomization_file_path, "r") as f:
            randomization_log = json.load(f)
        randomize_scene_from_log(controller, randomization_log)
        print("Loaded Randomization from {}".format(randomization_file_path))
    else:
        randomization_log = randomize_scene(args, controller)
        with open(randomization_file_path, "w") as f:
            json.dump(randomization_log, f)
        print("Created randomization and saved to {}".format(randomization_file_path))
            
    return randomization_log

def main(args: argparse.Namespace):
    save_folder_name = (
        args.scene_name
        if args.save_suffix is None
        else args.scene_name + "_" + args.save_suffix
    )
    save_root = args.dataset_root + "/" + save_folder_name + "/"
    os.makedirs(save_root, exist_ok=True)

    args.save_folder_name = save_folder_name
    args.save_root = save_root

    # Initialize the controller
    controller = Controller(
        agentMode="default",
        visibilityDistance=1.5,
        scene=get_scene(args.scene_name),
        # step sizes
        gridSize=args.grid_size,
        snapToGrid=False,
        rotateStepDegrees=90,
        # image modalities
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        renderSemanticSegmentation=True,
        # camera properties
        width=args.width,
        height=args.height,
        fieldOfView=args.fov,
        platform=CloudRendering,
    )

    load_or_randomize_scene(args, controller)
    
    # Save the reachable positions of the scene to a file
    reachable_positions = controller.step(
        action="GetReachablePositions", raise_for_failure=True
    ).metadata["actionReturn"]
    reachable_file = save_root + "/reachable.json"
    with open(reachable_file, "w") as f:
        json.dump(reachable_positions, f)

    # Get the poses to generate observations
    if args.sample_method == "random":
        sampled_poses = sample_pose_random(controller, args.n_sample)
    elif args.sample_method == "uniform":
        sampled_poses = sample_pose_uniform(controller, args.n_sample)
    elif args.sample_method == "from_file":
        traj_file = save_root + "/" + args.traj_file_name
        sampled_poses = sample_pose_from_file(traj_file)
        save_root = os.path.dirname(traj_file)
    else:
        raise ValueError("Unknown sample method: {}".format(args.sample_method))

    # Capture and save the top-down frame
    top_down_frame, top_down_grid = get_top_down_frame(controller)
    top_down_path = save_root + "/top_down.png"
    top_down_frame.save(top_down_path)
    top_down_path = save_root + "/top_down_grid.png"
    top_down_grid.save(top_down_path)
    
    if args.topdown_only:
        exit(0)

    # Generate the images according to the trajectory and save them
    K = compute_intrinsics(args.fov, args.height, args.width)
    generate_obs_from_poses(
        controller=controller,
        K=K,
        sampled_poses=sampled_poses,
        save_root=save_root,
        depth_scale=args.depth_scale,
    )


def main_interact(args: argparse.Namespace):
    '''
    Interact with the AI2Thor simulator, navigating the robot. 
    The agent trajectory will be saved to a file as a file. 
    Note that this saves the agent pose but not the camera pose. 
    '''
    save_folder_name = (
        args.scene_name + "_interact"
        if args.save_suffix is None
        else args.scene_name + "_" + args.save_suffix
    )
    save_root = args.dataset_root + "/" + save_folder_name + "/"
    os.makedirs(save_root, exist_ok=True)
    
    args.save_folder_name = save_folder_name
    args.save_root = save_root
    
    grid_size = 0.05
    rot_step = 2

    controller = Controller(
        gridSize=grid_size,
        rotateStepDegrees=rot_step,
        snapToGrid=False,
        scene=get_scene(args.scene_name),
        # camera properties
        width=args.width,
        height=args.height,
        fieldOfView=args.fov,
    )

    load_or_randomize_scene(args, controller)
    
    controller.step(
        action="LookUp",
        degrees=30
    )
    
    agent_logs = controller.interact()

    print("len(agent_logs):", len(agent_logs))

    trajectory_logs = {
        "scene_name": args.scene_name,
        "grid_size": grid_size,
        "rot_step": rot_step,
        "fov": args.fov,
        "height": args.height,
        "width": args.width,
        "agent_logs": agent_logs,
    }

    # Save log into a json file
    if not args.no_save:
        log_path = save_root + "/" + args.traj_file_name
        print("Saving interaction log to: ", log_path)
        with open(log_path, "w") as f:
            json.dump(trajectory_logs, f, indent=2)
            
def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Program Arguments")
    parser.add_argument(
        "--dataset_root",
        default=str(Path("~/ldata/ai2thor/").expanduser()),
        help="The root path to the dataset.",
    )
    parser.add_argument(
        "--grid_size",
        default=0.5,
        type=float,
        help="The translational step size in the scene (default 0.25).",
    )
    
    parser.add_argument(
        "--interact", action="store_true", help="Run in interactive mode. Requires GUI access."
    )
    parser.add_argument(
        "--traj_file_name", type=str, default="trajectory.json", 
        help="The name of the trajectory file to load."
    )
    
    parser.add_argument(
        "--no_save", action="store_true", help="Do not save trajectories from the interaction."
    )
    parser.add_argument(
        "--height", default=480, type=int, help="The height of the image."
    )
    parser.add_argument(
        "--width", default=640, type=int, help="The width of the image."
    )
    parser.add_argument(
        "--fov", default=90, type=int, help="The (vertical) field of view of the camera."
    )
    parser.add_argument(
        "--save_video", action="store_true", help="Save the video of the generated RGB frames."
    )
    
    parser.add_argument("--scene_name", default="train_3")
    parser.add_argument("--save_suffix", default=None)
    parser.add_argument("--randomize_lighting", action="store_true")
    parser.add_argument("--randomize_material", action="store_true")

    # Randomly remove objects in the scene
    parser.add_argument(
        "--randomize_remove_ratio",
        default=0.0,
        type=float,
        help="The probability to remove any object in the scene (0.0 - 1.0)",
    )
    parser.add_argument(
        "--randomize_remove_level", 
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="""What kind of objects to remove duing randomization 
        1: all objects except those in NOT_TO_REMOVE; 
        2: objects that are pickupable or moveable;
        3: objects that are pickupable""",
    )
    
    # Randomly moving objects in the scene
    parser.add_argument(
        "--randomize_move_pickupable_ratio",
        default=0.0,
        type=float,
        help="The ratio of pickupable objects to move.",
    )
    parser.add_argument(
        "--randomize_move_moveable_ratio",
        default=0.0,
        type=float,
        help="The ratio of moveable objects to move.",
    )
    
    parser.add_argument(
        "--topdown_only", action="store_true", help="Generate and save only the topdown view."
    )
    parser.add_argument(
        "--depth_scale", default=1000.0, type=float, help="The scale of the depth."
    )
    parser.add_argument(
        "--n_sample",
        default=-1,
        type=int,
        help="The number of images to generate. (-1 means all reachable positions are sampled)",
    )
    parser.add_argument(
        "--sample_method",
        default="uniform",
        choices=["random", "uniform", "from_file"],
        help="The method to sample the poses (random, uniform, from_file).",
    )
    parser.add_argument("--seed", default=0, type=int)
    
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Set up random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.interact:
        main_interact(args)
    else:
        main(args)
