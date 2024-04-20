import logging

from conceptgraph.utils.general_utils import find_existing_image_path
from conceptgraph.utils.geometry import rotation_matrix_to_quaternion
import numpy as np

class OptionalReRun:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config_use_rerun = None
            cls._instance._rerun = None
        return cls._instance

    def set_use_rerun(self, config_use_rerun):
        self._config_use_rerun = config_use_rerun
        if self._config_use_rerun and self._rerun is None:
            try:
                import rerun as rr
                self._rerun = rr
                logging.info("rerun is installed. Using rerun for logging.")
            except ImportError:
                logging.info("rerun is not installed. Not using rerun for logging.")
        else:
            logging.info("rerun functionality is disabled in the config. Not using rerun for logging.")

    def __getattr__(self, name):
        def method(*args, **kwargs):
            if self._config_use_rerun and self._rerun:
                func = getattr(self._rerun, name, None)
                if func:
                    return func(*args, **kwargs)
                else:
                    logging.debug(f"'{name}' is not a valid rerun method.")
            else:
                if not self._config_use_rerun:
                    logging.debug(f"Skipping optional rerun call to '{name}' because rerun usage is disabled.")
                elif self._rerun is None:
                    logging.debug(f"Skipping optional rerun call to '{name}' because rerun is not installed.")
        return method

# basically the import statement 
orr = OptionalReRun()

def orr_log_camera(intrinsics, adjusted_pose, prev_adjusted_pose, img_width, img_height, frame_idx):
    # Extract intrinsic camera parameters
    focal_length = [intrinsics[0, 0].item(), intrinsics[1, 1].item()]
    principal_point = [intrinsics[0, 2].item(), intrinsics[1, 2].item()]
    resolution = [img_width, img_height]  # Width x Height from the RGB image

    # Log camera intrinsics and resolution
    orr.log(
        "world/camera",
        orr.Pinhole(
            resolution=resolution,
            focal_length=focal_length,
            principal_point=principal_point,
        )
    )

    # Convert the current adjusted pose to translation and quaternion for logging
    translation = adjusted_pose[:3, 3].tolist()
    quaternion = rotation_matrix_to_quaternion(adjusted_pose[:3, :3])
    orr.log(
        "world/camera",
        orr.Transform3D(translation=translation, rotation=quaternion, from_parent=False)
    )

    # Log trajectory if not the first frame
    if frame_idx != 0:
        prev_translation = prev_adjusted_pose[:3, 3].tolist()
        prev_quaternion = rotation_matrix_to_quaternion(prev_adjusted_pose[:3, :3])

        # Log a line strip from the previous to the current camera pose
        orr.log(
            f"world/camera_trajectory/{frame_idx}",
            orr.LineStrips3D(
                [np.vstack([prev_translation, translation]).tolist()],
                colors=[[255, 0, 0]]  # Red color for the trajectory line
            )
        )
    prev_adjusted_pose = adjusted_pose.copy()
    return prev_adjusted_pose
        
def orr_log_rgb_image(color_path):
    # Log RGB image from the specified path
    color_path = color_path
    orr.log(
        "world/camera/rgb_image_encoded",
        orr.ImageEncoded(path=str(color_path))
    )
    
def orr_log_depth_image(depth_tensor):

    depth_in_meters = depth_tensor.numpy() 

    # Ensure depth data is in the expected format for rerun (HxW)
    # depth_in_meters should be a 2D numpy array at this point
    assert len(depth_in_meters.shape) == 2, "Depth data must be a 2D array"

    # This should really use meter = 1.0, but setting it to that makes it too big
    # I wanna confirm its not me before making an issue on their github
    orr.log(
        "world/camera/depth",
        orr.DepthImage(depth_in_meters , meter=0.9999999)
    )

def orr_log_annotated_image(color_path, det_exp_vis_path):
    # Check if the visualizations exist and log them
    color_path = color_path
    det_exp_vis_path = det_exp_vis_path
    base_vis_save_path = det_exp_vis_path / color_path.stem
    existing_vis_save_path = find_existing_image_path(base_vis_save_path, ['.jpg', '.png'])
    if existing_vis_save_path:
        orr.log(
            "world/camera/rgb_image_annotated",
            orr.ImageEncoded(path=existing_vis_save_path)
        )

def orr_log_objs_pcd_and_bbox(objects, obj_classes):
        
    for obj_idx, obj in enumerate(objects):
                
        if obj['num_detections'] < 3:
            continue

        if obj['is_background']:
            continue
        
        obj_label = f"{obj['curr_obj_num']}_{obj['class_name']}"
        obj_label = obj_label.replace(" ", "_")
        base_entity_path = "world/objects"
        entity_path = f"world/objects/{obj_label}"

        # Convert points and colors to NumPy arrays
        positions = np.asarray(obj['pcd'].points)
        if hasattr(obj['pcd'], 'colors') and len(obj['pcd'].colors) > 0:
            colors = np.asarray(obj['pcd'].colors) * 255
            # make them ints
            colors = colors.astype(np.uint8)
        else:
            colors = None
            
        curr_obj_color = obj_classes.get_class_color(obj['class_name'])
        curr_obj_inst_color = obj['inst_color']

        # Log point cloud data
        orr.log(
            base_entity_path + "/rgb_pcd" + f"/{obj_label}",
            # entity_path + "/pcd", 
            orr.Points3D(
                positions, 
                colors=colors,
                # labels=[obj_label],
            ),
            orr.AnyValues(
                uuid = str(obj['id']),
            )
        )
        
        # Log point cloud data
        orr.log(
            base_entity_path + "/seg_pcd" + f"/{obj_label}",
            # entity_path + "/pcd", 
            orr.Points3D(
                positions, 
                colors=[curr_obj_color],
                # labels=[obj_label],
            ),
            orr.AnyValues(
                uuid = str(obj['id']),
            )
        )
        
        # Log point cloud data
        orr.log(
            base_entity_path + "/inst_pcd" + f"/{obj_label}",
            # entity_path + "/pcd", 
            orr.Points3D(
                positions, 
                colors=curr_obj_inst_color,
                # labels=[obj_label],
            ),
            orr.AnyValues(
                uuid = str(obj['id']),
            )
        )

        # Assuming bbox is extracted as before
        bbox = obj['bbox']
        centers = [bbox.center]
        half_sizes = [bbox.extent /2 ]
        # Convert rotation matrix to quaternion
        bbox_quaternion = [rotation_matrix_to_quaternion(bbox.R)]

        orr.log(
            base_entity_path + "/bbox" + f"/{obj_label}",
            # entity_path + "/bbox", 
            orr.Boxes3D(
                centers=centers, 
                half_sizes=half_sizes, 
                rotations=bbox_quaternion,
                colors=[curr_obj_color],
                # labels=[f"{obj_label}_({obj['num_detections']})"],
            ),
            orr.AnyValues(
                uuid = str(obj['id']),
            )
        )
        
        orr.log(
            base_entity_path + "/bbox_w_labels" + f"/{obj_label}",
            # entity_path + "/bbox", 
            orr.Boxes3D(
                centers=centers, 
                half_sizes=half_sizes, 
                rotations=bbox_quaternion,
                labels=[f"{obj_label}_({obj['num_detections']})"],
                colors=[curr_obj_color],
            ),
            orr.AnyValues(
                uuid = str(obj['id']),
            )
        )
        # {obj['class_name']}
        orr.log(
            base_entity_path + "/bbox_w_name" + f"/{obj_label}",
            # entity_path + "/bbox", 
            orr.Boxes3D(
                centers=centers, 
                half_sizes=half_sizes, 
                rotations=bbox_quaternion,
                labels=[f"{obj['class_name']}"],
                colors=[curr_obj_color],
            ),
            orr.AnyValues(
                uuid = str(obj['id']),
            )
        )
        
def orr_log_edges(objects, map_edges, obj_classes):
     # do the same for edges
    for map_edge_tuple in map_edges.edges_by_index.items():
        obj1_idx, obj2_idx = map_edge_tuple[0]
        map_edge = map_edge_tuple[1]
        num_dets = map_edge.num_detections
        if num_dets < 3:
            continue
        obj1_label = f"{objects[obj1_idx]['curr_obj_num']}"
        obj2_label = f"{objects[obj2_idx]['curr_obj_num']}"
        
        obj_1_num_dets = objects[obj1_idx]['num_detections']
        obj_2_num_dets = objects[obj2_idx]['num_detections']
        
        if obj_1_num_dets < 3 or obj_2_num_dets < 3:
            continue
        
        
        rel_type = map_edge.rel_type.replace(" ", "_")
        edge_label_by_curr_num = f"{obj1_label}_{rel_type}_{obj2_label}"
        entity_path = f"world/edges/{edge_label_by_curr_num}"
        base_entity_path = "world/edges"
        
        
        endpoints = map_edges.get_edge_endpoints(obj1_idx, obj2_idx)
        obj1_full_label = f"{objects[obj1_idx]['curr_obj_num']}_{objects[obj1_idx]['class_name']}".replace(" ", "_")
        obj2_full_label = f"{objects[obj2_idx]['curr_obj_num']}_{objects[obj2_idx]['class_name']}".replace(" ", "_")
        full_label = f"{obj1_full_label}__{rel_type}__{obj2_full_label}_({num_dets})"
        name_label = f"{objects[obj1_idx]['class_name']}__{rel_type}__{objects[obj2_idx]['class_name']}"
        
        obj_2_color = obj_classes.get_class_color(objects[obj2_idx]['class_name'])
        orr.log(
            base_entity_path + f"/edges_no_labels" + f"/{edge_label_by_curr_num}", 
            orr.LineStrips3D(
                endpoints,
                colors=[obj_2_color],
                # labels=[f"{num_dets}"],
            ),
            orr.AnyValues(
                full_label = full_label
            )
        )
        
        orr.log(
            base_entity_path + f"/edges_w_num_det_labels" + f"/{edge_label_by_curr_num}", 
            orr.LineStrips3D(
                endpoints,
                labels=[f"{num_dets}"],
                colors=[obj_2_color],
            ),
            orr.AnyValues(
                full_label = full_label
            )
        )
        
        orr.log(
            base_entity_path + f"/edges_w_rel_type_labels" + f"/{edge_label_by_curr_num}", 
            orr.LineStrips3D(
                endpoints,
                labels=[f"{rel_type}"],
                colors=[obj_2_color],
            ),
            orr.AnyValues(
                full_label = full_label
            )
        )
        
        orr.log(
            base_entity_path + f"/edges_w_full_labels" + f"/{edge_label_by_curr_num}", 
            orr.LineStrips3D(
                endpoints,
                labels=[f"{full_label}"],
                colors=[obj_2_color],
            ),
            orr.AnyValues(
                full_label = full_label
            )
        )
        
        orr.log(
            base_entity_path + f"/edges_w_names" + f"/{edge_label_by_curr_num}", 
            orr.LineStrips3D(
                endpoints,
                labels=[f"{name_label}"],
                colors=[obj_2_color],
            ),
            orr.AnyValues(
                full_label = full_label
            )
        )