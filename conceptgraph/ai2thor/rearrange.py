import copy
import random

import numpy as np
from ai2thor.controller import Controller

from conceptgraph.ai2thor.utils import parse_object_receptacle_mapping
from conceptgraph.utils.ai2thor import compute_pose, compute_posrot

def rearrange_objects(
    controller: Controller,
    pickupable_move_ratio: float,
    moveable_move_ratio: float,
    random_seed: int | None = None,
    reset: bool = False,
):
    '''
    Rearrange objects in the scene. 
    '''
    if reset:
        controller.reset()
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Get the initial objects in the scene
    objects_init = controller.last_event.metadata["objects"]
    
    starting_poses = []
    for obj in objects_init:
        if obj["pickupable"] or obj["moveable"]:
            starting_poses.append({
                "objectId": obj["objectId"],
                "name": obj['name'],
                "objectName": obj['name'],
                "position": obj["position"],
                "rotation": obj["rotation"],
            })
    
    obj2receptacle, receptacle2obj = parse_object_receptacle_mapping(controller)
    
    pickupable_objects_init = [o for o in objects_init if o["pickupable"]]
    moveable_objects = [o for o in objects_init if o["moveable"] and not o["pickupable"]]
    
    n_move_pickupable = int(len(pickupable_objects_init) * pickupable_move_ratio)
    n_move_moveable = int(len(moveable_objects) * moveable_move_ratio)
    
    print(f"Try moving {n_move_pickupable}/{len(pickupable_objects_init)} pickupable objects and {n_move_moveable}/{len(moveable_objects)} moveable objects")
    
    # First random move the movable objects
    moveable_objects = np.random.permutation(moveable_objects)
    n_moved_moveable = 0
    idx_obj_tried = -1
    while n_moved_moveable < n_move_moveable:
        idx_obj_tried += 1
        obj = moveable_objects[idx_obj_tried]
        obj_id = obj["objectId"]
        
        if "Television" in obj_id:
            # TVs do not fall to the ground
            continue
        
        print("Try moving object", obj["objectId"], end="...")
        
        original_pos = copy.deepcopy(obj["position"])
        original_rot = copy.deepcopy(obj["rotation"])
        original_pose = compute_pose(original_pos, original_rot)

        reachable_positions = controller.step(
            action="GetReachablePositions", raise_for_failure=True
        ).metadata["actionReturn"]
        
        # Try teleporting the object to a random reachable position
        # Retry a few times if the teleport fails
        teleport_success = False
        try_teleport_count = 0
        while not teleport_success:
            target_pos = random.choice(reachable_positions)
            target_rot = copy.deepcopy(obj["rotation"])
            
            # The height should be the same as the original object
            target_pos['y'] = original_pos['y']

            # Randomize the rotation
            target_rot['y'] = random.random() * 360
            
            event = controller.step(
                "TeleportObject",
                objectId=obj["objectId"],
                position=target_pos,
                rotation=target_rot,
                makeUnbreakable=True,
            )
            
            if event.metadata['lastActionSuccess']:
                teleport_success = True
                break

            try_teleport_count += 1
            if try_teleport_count > 30:
                break
            
        if not teleport_success:
            print("Failed")
            continue
            
        # Also teleport the associate objects that are contained by this object
        target_pose = compute_pose(target_pos, target_rot)
        if obj_id in receptacle2obj:
            relative_pose = target_pose @ np.linalg.inv(original_pose)
            for o_id in receptacle2obj[obj_id]:
                o = next(_ for _ in controller.last_event.metadata["objects"] if _['objectId'] == o_id)
                o_pose_old = compute_pose(o["position"], o["rotation"])
                o_pose_new = relative_pose @ o_pose_old
                o_pos_new, o_rot_new = compute_posrot(o_pose_new)
                o_pos_new['y'] += 0.1 # To avoid collision
                event = controller.step(
                    "TeleportObject",
                    objectId=o["objectId"],
                    position=o_pos_new,
                    rotation=o_rot_new,
                    makeUnbreakable=True,
                )
                if not event.metadata['lastActionSuccess']:
                    print(event.metadata['errorMessage'])
                    print(f"Failed to teleport {o['objectId']} contained by {obj_id}")
        
        for _ in range(12):
            # to let physics settle.
            controller.step("Pass")
            
        # print(original_pos, target_pos, new_pos)
            
        print("Success")
        n_moved_moveable += 1
        
        if idx_obj_tried >= len(moveable_objects):
            if n_moved_moveable < n_move_moveable:
                print("Did not move enough moveable objects in the scene")
            break
        
    # Then let's try moving the pickupable objects
    pickupable_objects_init = np.random.permutation(pickupable_objects_init)
    if len(pickupable_objects_init) > 0:
        pickupable_objects_to_move = pickupable_objects_init[:n_move_pickupable]
        pickupable_objects_to_stay = pickupable_objects_init[n_move_pickupable:]
        # Seems this does not change the object id? 
        event = controller.step(
            action="InitialRandomSpawn",
            randomSeed=0 if random_seed is None else random_seed,
            forceVisible=True,
            numPlacementAttempts=5,
            placeStationary=True,
            excludedObjectIds=[o["objectId"] for o in pickupable_objects_to_stay],
        )
        
        if event.metadata['lastActionSuccess']:
            print("Successfully moved pickupable objects")
    
    # Let physics settle
    for _ in range(12):
        controller.step("Pass")
    
    # Record the pose of the final objects in the scene
    target_poses = []
    for obj in controller.last_event.metadata["objects"]:
        if obj['pickupable'] or obj['moveable']:
            target_poses.append({
                "objectId": obj["objectId"],
                "name": obj['name'],
                "objectName": obj['name'],
                "position": obj["position"],
                "rotation": obj["rotation"],
            })
            
    # Assert no object name changed
    starting_poses.sort(key=lambda x: x['name'])
    target_poses.sort(key=lambda x: x['name'])
    for sp, tp in zip(starting_poses, target_poses):
        if sp['name'] != tp['name'] or \
           sp['objectId'] != tp['objectId'] or \
           sp['objectName'] != tp['objectName']:
            print("object name changed:", sp['name'], tp['name'])
            import pdb; pdb.set_trace()
        
    return starting_poses, target_poses