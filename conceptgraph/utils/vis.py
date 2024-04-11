import copy
import logging
from pathlib import Path
from typing import Iterable
import dataclasses
from PIL import Image
import cv2

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import open3d as o3d

import supervision as sv
from supervision.draw.color import Color, ColorPalette
from conceptgraph.slam.slam_classes import MapObjectList

from typing import List, Optional, Union


class OnlineObjectRenderer():
    '''
    Refactor of the open3d visualization code to make it more modular
    '''
    def __init__(
        self, 
        view_param: str | dict,
        base_objects: MapObjectList | None = None,
        gray_map: bool = False
    ) -> None:
        # If the base objects are provided, we will visualize them
        if base_objects is not None:
            self.n_base_objects = len(base_objects)

            base_pcds_vis = copy.deepcopy(base_objects.get_values("pcd"))
            base_bboxes_vis = copy.deepcopy(base_objects.get_values("bbox"))
            for i in range(self.n_base_objects):
                base_pcds_vis[i] = base_pcds_vis[i].voxel_down_sample(voxel_size=0.08)
                if gray_map:
                    base_pcds_vis[i].paint_uniform_color([0.5, 0.5, 0.5])
            for i in range(self.n_base_objects):
                base_bboxes_vis[i].color = [0.5, 0.5, 0.5]
            
            self.base_pcds_vis = base_pcds_vis
            self.base_bboxes_vis = base_bboxes_vis
        else:
            self.n_base_objects = 0
        
        self.est_traj = []
        self.gt_traj = []
        
        self.cmap = matplotlib.colormaps.get_cmap("turbo")

        if isinstance(view_param, str):
            self.view_param = o3d.io.read_pinhole_camera_parameters(view_param)
        else:
            self.view_param = view_param
            
        self.window_height = self.view_param.intrinsic.height
        self.window_width = self.view_param.intrinsic.width
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            width = self.window_width,
            height = self.window_height,
        )
        
        self.vis_ctrl = self.vis.get_view_control()
        self.vis_ctrl.convert_from_pinhole_camera_parameters(self.view_param)
        
    def filter_base_by_mask(self, mask: Iterable[bool]):
        assert len(mask) == self.n_base_objects
        self.base_pcds_vis = [pcd for pcd, m in zip(self.base_pcds_vis, mask) if m]
        self.base_bboxes_vis = [bbox for bbox, m in zip(self.base_bboxes_vis, mask) if m]
        self.n_base_objects = len(self.base_pcds_vis)
    
    def step(
        self,
        image: Image.Image,
        pcds: list[o3d.geometry.PointCloud] | None = None,
        pcd_colors: np.ndarray | None = None,
        est_pose: np.ndarray | None = None,
        gt_pose: np.ndarray | None = None,
        base_objects_color: dict | None = None,
        new_objects: MapObjectList = None,
        paint_new_objects: bool = True,
        return_vis_handle: bool = False,
    ):
        # Remove all the geometries
        self.vis.clear_geometries()
        
        # Add the pose cameras and trajectories
        if est_pose is not None:
            self.est_traj.append(est_pose)
            est_camera_frustum = better_camera_frustum(
                est_pose, image.height, image.width, scale=0.5, color=[1., 0, 0]
            )
            self.vis.add_geometry(est_camera_frustum)
            if len(self.est_traj) > 1:
                est_traj_lineset = poses2lineset(np.stack(self.est_traj), color=[1., 0, 0])
                self.vis.add_geometry(est_traj_lineset)
            
        if gt_pose is not None:
            self.gt_traj.append(gt_pose)
            gt_camera_frustum = better_camera_frustum(
                gt_pose, image.height, image.width, scale=0.5, color=[0, 1., 0]
            )
            self.vis.add_geometry(gt_camera_frustum)
            if len(self.gt_traj) > 1:
                gt_traj_lineset = poses2lineset(np.stack(self.gt_traj), color=[0, 1., 0])
                self.vis.add_geometry(gt_traj_lineset)
    
        # Add the base objects
        if self.n_base_objects > 0:
            if base_objects_color is not None:
                for obj_id in range(self.n_base_objects):
                    color = base_objects_color[obj_id]
                    self.base_pcds_vis[obj_id].paint_uniform_color(color)
                    self.base_bboxes_vis[obj_id].color = color
            
            for geom in self.base_pcds_vis + self.base_bboxes_vis:
                self.vis.add_geometry(geom)
            
        # Show the extra pcds to visualize
        if pcds is not None:
            for i in range(len(pcds)):
                pcds[i].transform(est_pose)
                if pcd_colors is not None:
                    pcds[i].paint_uniform_color(pcd_colors[i][:3])
                self.vis.add_geometry(pcds[i])
            
        # Show the extra new objects
        if new_objects is not None:
            for obj in new_objects:
                pcd = copy.deepcopy(obj['pcd'])
                bbox = copy.deepcopy(obj['bbox'])
                bbox.color = [0.0, 0.0, 1.0]
                if paint_new_objects:
                    pcd.paint_uniform_color([0.0, 1.0, 0.0])
                    bbox.color = [0.0, 1.0, 0.0]
                
                self.vis.add_geometry(pcd)
                self.vis.add_geometry(bbox)
        
        self.vis_ctrl.convert_from_pinhole_camera_parameters(self.view_param)
        
        self.vis.poll_events()
        self.vis.update_renderer()
        
        rendered_image = self.vis.capture_screen_float_buffer(False)
        rendered_image = np.asarray(rendered_image)
        
        if return_vis_handle:
            return rendered_image, self.vis
        else:
            return rendered_image, None

def get_random_colors(num_colors):
    '''
    Generate random colors for visualization
    
    Args:
        num_colors (int): number of colors to generate
        
    Returns:
        colors (np.ndarray): (num_colors, 3) array of colors, in RGB, [0, 1]
    '''
    colors = []
    for i in range(num_colors):
        colors.append(np.random.rand(3))
    colors = np.array(colors)
    return colors

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax, label=None):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
    
    if label is not None:
        ax.text(x0, y0, label)
        
def vis_result_fast_on_depth(
    depth_image: np.ndarray, 
    detections: sv.Detections, 
    classes: list[str], 
    color: Color | ColorPalette = ColorPalette.default(), 
    instance_random_color: bool = False,
    draw_bbox: bool = True,
) -> np.ndarray:
    '''
    Annotate the image with the detection results. 
    This is fast but of the same resolution of the input image, thus can be blurry. 
    '''
    # annotate image with detections
    box_annotator = sv.BoxAnnotator(
        color = color,
        text_scale=0.3,
        text_thickness=1,
        text_padding=2,
    )
    mask_annotator = sv.MaskAnnotator(
        color = color,
        opacity=0.2,
    )

    if hasattr(detections, 'confidence') and hasattr(detections, 'class_id'):
        confidences = detections.confidence
        class_ids = detections.class_id
        if confidences is not None:
            labels = [
                f"{classes[class_id]} {confidence:0.2f}"
                for confidence, class_id in zip(confidences, class_ids)
            ]
        else:
            labels = [f"{classes[class_id]}" for class_id in class_ids]
    else:
        print("Detections object does not have 'confidence' or 'class_id' attributes or one of them is missing.")

    
    if instance_random_color:
        # generate random colors for each segmentation
        # First create a shallow copy of the input detections
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))
        
    annotated_image = mask_annotator.annotate(scene=depth_image.copy(), detections=detections)
    
    if draw_bbox:
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image, labels

def filter_detections(
    detections: sv.Detections, 
    classes: list[str], 
    top_x_detections: Optional[int] = None, 
    confidence_threshold: float = 0.0,
    given_labels: Optional[list[str]] = None
) -> tuple[sv.Detections, list[str]]:
    '''
    Filter detections based on confidence threshold and top X detections.
    Returns a tuple of filtered detections and labels.
    '''
    temp = given_labels
    if not (hasattr(detections, 'confidence') and hasattr(detections, 'class_id') and hasattr(detections, 'xyxy')):
        print("Detections object is missing required attributes.")
        return detections, []

    detections_combined = []
    for index, (confidence, class_id, xyxy) in enumerate(zip(detections.confidence, detections.class_id, detections.xyxy)):
        if confidence >= confidence_threshold:
            detections_combined.append((confidence, class_id, xyxy, index))

    # Sort by confidence in descending order
    detections_combined.sort(key=lambda x: x[0], reverse=True)

    if top_x_detections is not None:
        detections_combined = detections_combined[:top_x_detections]

    detections_combined = list(reversed(detections_combined))

    filtered_class_ids, filtered_confidences, filtered_xyxy, filtered_labels = [], [], [], []
    for det_idx, (confidence, class_id, xyxy, original_index) in enumerate(detections_combined):
        filtered_class_ids.append(class_id)
        filtered_confidences.append(confidence)
        filtered_xyxy.append(xyxy)
        # labels.append(f"{classes[class_id]} {confidence:0.2f}")
        # labels.append(f"{classes[class_id]} {det_idx + 1}")
        filtered_labels.append(given_labels[original_index])

    filtered_detections = sv.Detections(
        class_id=np.array(filtered_class_ids),
        confidence=np.array(filtered_confidences),
        xyxy=np.array(filtered_xyxy)
    )

    return filtered_detections, filtered_labels

class CustomBoxAnnotator(sv.BoxAnnotator):
    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.DEFAULT,
        thickness: int = 2,
        text_color: Color = Color.BLACK,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        detections: sv.Detections = None,
        labels: List[str] = None,
        save_path: str | None = None,
    ):
        super().__init__(color, thickness, text_color, text_scale, text_thickness, text_padding)
        self.label_mask = None
        self.save_path = save_path
        if self.save_path is not None:
            self.save_path.mkdir(parents=True, exist_ok=True)
            self.save_counter = 0
        self.curr_detections = detections
        self.labels = labels
        self.curr_box_idx = None
        self.curr_label_idx = None

    def reset_label_mask(self, image_shape):
        self.label_mask = np.zeros(image_shape[:2], dtype=np.bool_)

    def is_area_occupied(self, x1, y1, x2, y2):
        if np.any(self.label_mask[y1:y2, x1:x2]):
            return True
        return False

    def mark_area_as_occupied(self, x1, y1, x2, y2):
        self.label_mask[y1:y2, x1:x2] = True
        
    def save_debug_image(self, scene):
        if self.save_path is not None:
            save_file = self.save_path / f"step_{self.save_counter}.jpg"
            cv2.imwrite(str(save_file), scene)
            self.save_counter += 1

    def find_label_position(self, x1, y1, x2, y2, text_width, text_height, image_height, image_width):
        '''
        Find the best position for the label text given the bounding box coordinates and text dimensions.
        Check for collisions with existing labels and image boundaries.
        '''
        padding = self.text_padding * 2
        text_full_width = text_width + padding
        text_full_height = text_height + padding
        
        # Initialize naive positions as a list, starting from 'top_left' and moving clockwise
        naive_positions = [
            {"name": "top_left", "x1": x1, "y1": y1 - text_full_height, "x2": x1 + text_full_width, "y2": y1},
            {"name": "top_right", "x1": x2 - text_full_width, "y1": y1 - text_full_height, "x2": x2, "y2": y1},
            {"name": "bottom_right", "x1": x2 - text_full_width, "y1": y2, "x2": x2, "y2": y2 + text_full_height},
            {"name": "bottom_left", "x1": x1, "y1": y2, "x2": x1 + text_full_width, "y2": y2 + text_full_height},
        ]
        
        # Check each image boundary and adjust if necessary
        positions_list = []
        for curr_label_pos in naive_positions:
            # Left iamge boundarry check
            if curr_label_pos["x1"] < 0:  
                curr_label_pos["x1"] = 0
                curr_label_pos["x2"] = min(text_full_width, image_width)
            # Top image boundary check
            if curr_label_pos["y1"] < 0:
                curr_label_pos["y1"] = 0
                curr_label_pos["y2"] = min(text_full_height, image_height)
            # Right image boundary check
            if curr_label_pos["x2"] > image_width:
                overflow = curr_label_pos["x2"] - image_width
                curr_label_pos["x1"] = max(curr_label_pos["x1"] - overflow, 0)
                curr_label_pos["x2"] = image_width
            # Bottom image boundary check
            if curr_label_pos["y2"] > image_height:
                overflow = curr_label_pos["y2"] - image_height
                curr_label_pos["y1"] = max(curr_label_pos["y1"] - overflow, 0)
                curr_label_pos["y2"] = image_height
            positions_list.append(curr_label_pos)
        
        # Check for collisions with existing labels
        curr_id = self.curr_detections.class_id[self.curr_label_idx]
        curr_label = self.labels[self.curr_label_idx]
        curr_num = self.curr_label_idx + 1
        curr_total = len(self.curr_detections)
        for pos in positions_list:
            collision_flag = self.is_area_occupied(pos['x1'], pos['y1'], pos['x2'], pos['y2'])
            logging.debug(f"Checking label {curr_num}/{curr_total} with class ID: {curr_id}, NAME: {curr_label}")
            logging.debug(f"At {pos['name']} corner at ({pos['x1']}, {pos['y1']}, {pos['x2']}, {pos['y2']}).")
            logging.debug(f"Collision flag: {collision_flag}")
            if not collision_flag:
                return (pos['x1'], pos['y1'], pos['x2'], pos['y2'], pos['name'])

        # If all positions have been checked and all have collisions, log that no viable position was found
        logging.debug(f"NO VIABLE POSITION for label {curr_num}/{curr_total} with class ID: {curr_id}, NAME: {curr_label}")
        return (pos['x1'], pos['y1'], pos['x2'], pos['y2'], pos['name']) 

    def annotate(
        self,
        scene: np.ndarray,
        detections: sv.Detections,
        labels: Optional[List[str]] = None,
        skip_label: bool = False,
    ) -> np.ndarray:
        self.reset_label_mask(scene.shape)
        font = cv2.FONT_HERSHEY_SIMPLEX

        logging.debug("Starting bounding box drawing process.")
        for i, (x1, y1, x2, y2) in enumerate(detections.xyxy.astype(int)):
            self.curr_box_idx = i
            class_id = detections.class_id[i]
            text = labels[i] if labels is not None and len(labels) == len(detections) else str(class_id)
            color = self.color.by_idx(class_id) if isinstance(self.color, ColorPalette) else self.color
            cv2.rectangle(scene, (x1, y1), (x2, y2), color=color.as_bgr(), thickness=self.thickness)
            self.save_debug_image(scene)
            logging.debug(f"Bounding box {i+1}/{len(detections)} drawn for class ID: {class_id}, NAME: {text}")
            logging.debug(f"with color {str(color)} at {(x1, y1, x2, y2)}")
            
        if skip_label:
            return scene
        
        logging.debug("Starting label drawing process.")
        for i, (x1, y1, x2, y2) in enumerate(detections.xyxy.astype(int)):
            self.curr_label_idx = i
            class_id = detections.class_id[i]
            color = self.color.by_idx(class_id) if isinstance(self.color, ColorPalette) else self.color
            text = labels[i] if labels is not None and len(labels) == len(detections) else str(class_id)

            text_width, text_height = cv2.getTextSize(text, font, self.text_scale, self.text_thickness)[0]
            logging.debug(f"Calculated text dimensions for '{text}': width={text_width}, height={text_height}")

            label_x1, label_y1, label_x2, label_y2, pos_desc = self.find_label_position(x1, y1, x2, y2, text_width, text_height, scene.shape[0], scene.shape[1])

            self.mark_area_as_occupied(label_x1, label_y1, label_x2, label_y2)
            cv2.rectangle(scene, (label_x1, label_y1), (label_x2, label_y2), color=color.as_bgr(), thickness=cv2.FILLED)
            
            cv2.putText(scene, text, (int(label_x1) + self.text_padding, int(label_y2) - self.text_padding), font, self.text_scale, self.text_color.as_rgb(), self.text_thickness, cv2.LINE_AA)

            self.save_debug_image(scene)
            logging.debug(f"Label {i+1}/{len(detections)} for class ID: {class_id}, NAME: {text}")
            logging.debug(f"drawn with background color {str(color)} at {pos_desc} corner")
            logging.debug(f"with bounding box ({label_x1}, {label_y1}, {label_x2}, {label_y2})")

        return scene

def vis_result_for_vlm(
    image: np.ndarray, 
    detections: sv.Detections, 
    labels: list[str], 
    color: Color | ColorPalette = ColorPalette.default(), 
    draw_bbox: bool = True,
    thickness: int = 2,
    text_scale: float = 0.3,
    text_thickness: int = 1,
    text_padding: int = 2,
    save_path: Optional[str] = None
) -> np.ndarray:
    '''
    Annotate the image with the filtered detection results.
    '''

    save_path = Path(save_path) if save_path is not None else None

    box_annotator = CustomBoxAnnotator(
        color=color,
        thickness=thickness,
        text_scale=text_scale,
        text_thickness=text_thickness,
        text_padding=text_padding,
        detections=detections,
        labels=labels,
        save_path=save_path if save_path is not None else None,
    )

    annotated_image = image.copy()
    if draw_bbox:
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image, labels

        
def vis_result_fast(
    image: np.ndarray, 
    detections: sv.Detections, 
    classes: list[str], 
    color: Color | ColorPalette = ColorPalette.default(), 
    instance_random_color: bool = False,
    draw_bbox: bool = True,
) -> np.ndarray:
    '''
    Annotate the image with the detection results. 
    This is fast but of the same resolution of the input image, thus can be blurry. 
    '''
    # annotate image with detections
    box_annotator = sv.BoxAnnotator(
        color = color,
        text_scale=0.3,
        text_thickness=1,
        text_padding=2,
    )
    mask_annotator = sv.MaskAnnotator(
        color = color
    )

    if hasattr(detections, 'confidence') and hasattr(detections, 'class_id'):
        confidences = detections.confidence
        class_ids = detections.class_id
        if confidences is not None:
            labels = [
                f"{classes[class_id]} {confidence:0.2f}"
                for confidence, class_id in zip(confidences, class_ids)
            ]
        else:
            labels = [f"{classes[class_id]}" for class_id in class_ids]
    else:
        print("Detections object does not have 'confidence' or 'class_id' attributes or one of them is missing.")

    
    if instance_random_color:
        # generate random colors for each segmentation
        # First create a shallow copy of the input detections
        detections = dataclasses.replace(detections)
        detections.class_id = np.arange(len(detections))
        
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    
    if draw_bbox:
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image, labels

def vis_result_slow_caption(image, masks, boxes_filt, pred_phrases, caption, text_prompt):
    '''
    Annotate the image with detection results, together with captions and text prompts.
    This function is very slow, but the output is more readable.
    '''
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box, plt.gca(), label)

    plt.title('Tagging-Caption: ' + caption + '\n' + 'Tagging-classes: ' + text_prompt + '\n')
    plt.axis('off')
    
    # Convert the fig to a numpy array
    fig = plt.gcf()
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    return vis_image

def vis_sam_mask(anns):
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        
    return img

def poses2lineset(poses, color=[0, 0, 1]):
    '''
    Create a open3d line set from a batch of poses

    poses: (N, 4, 4)
    color: (3,)
    '''
    N = poses.shape[0]
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(poses[:, :3, 3])
    lineset.lines = o3d.utility.Vector2iVector(
        np.array([[i, i + 1] for i in range(N - 1)])
    )
    lineset.colors = o3d.utility.Vector3dVector([color for _ in range(len(lineset.lines))])
    return lineset

def create_camera_frustum(
    camera_pose, width=1, height=1, z_near=0.5, z_far=1, color=[0, 0, 1]
):
    K = np.array([[z_near, 0, 0], [0, z_near, 0], [0, 0, z_near + z_far]])
    points = np.array(
        [
            [-width / 2, -height / 2, z_near],
            [width / 2, -height / 2, z_near],
            [width / 2, height / 2, z_near],
            [-width / 2, height / 2, z_near],
            [0, 0, 0],
        ]
    )
    points_transformed = camera_pose[:3, :3] @ (K @ points.T) + camera_pose[:3, 3:4]
    points_transformed = points_transformed.T
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points_transformed)
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3]]
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return frustum


def better_camera_frustum(camera_pose, img_h, img_w, scale=3.0, color=[0, 0, 1]):
    # Convert camera pose tensor to numpy array
    if isinstance(camera_pose, torch.Tensor):
        camera_pose = camera_pose.numpy()
    
    # Define near and far distance (adjust these as needed)
    near = scale * 0.1
    far = scale * 1.0
    
    # Define frustum dimensions at the near plane (replace with appropriate values)
    frustum_h = near
    frustum_w = frustum_h * img_w / img_h  # Set frustum width based on its height and the image aspect ratio
    
    # Compute the 8 points that define the frustum
    points = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                u = x * (frustum_w // 2 if z == -1 else frustum_w * far / near)
                v = y * (frustum_h // 2 if z == -1 else frustum_h * far / near)
                d = near if z == -1 else far # Negate depth here
                # d = -near if z == -1 else -far # Negate depth here
                point = np.array([u, v, d, 1]).reshape(-1, 1)
                transformed_point = (camera_pose @ point).ravel()[:3]
                # transformed_point[0] *= -1  # Flip X-coordinate
                points.append(transformed_point) # Using camera pose directly
                # points.append((camera_pose_np @ point).ravel()[:3]) # Using camera pose directly
    
    # Create lines that connect the 8 points
    lines = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [3, 7], [2, 6]]
    
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])

    return frustum


# Copied from https://github.com/isl-org/Open3D/pull/738
def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

def save_video_detections(exp_out_path, save_path=None, fps=30):
    '''
    Save the detections in the folder as a video
    '''
    if save_path is None:
        save_path = exp_out_path / "vis_video.mp4"
    
    # Get the list of images
    image_files = list((exp_out_path / "vis").glob("*.jpg"))
    image_files.sort()
    
    # Read the first image to get the size
    image = Image.open(image_files[0])
    width, height = image.size
    
    # Create the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
    
    # Write the images to the video
    for image_file in image_files:
        image = cv2.imread(str(image_file))
        out.write(image)
    
    out.release()
    print(f"Video saved at {save_path}")

def save_video_from_frames(frames, save_path, fps=30):
    """
    Save a video from an array of frames.

    Args:
    - frames: An array of frames, each frame being an image array.
    - save_path: Path where the video should be saved.
    - fps: Frames per second for the output video.
    """
    # Ensure frames are in uint8
    frames = np.asarray(frames).astype(np.uint8)
    
    # Check if frames array is empty
    if frames.size == 0:
        print("No frames to save.")
        return
    
    # Get the size of the first frame
    height, width = frames[0].shape[:2]
    
    # Create the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
    
    # Write the frames to the video
    for frame in frames:
        # OpenCV expects BGR format, might need to convert from RGB to BGR
        if frame.shape[2] == 3:  # If frame has three channels
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    # Release the video writer
    out.release()
    print(f"Video saved at {save_path}")


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                # cylinder_segment = cylinder_segment.rotate(
                #     R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a))
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a))
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder, reset_bounding_box=False)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder, reset_bounding_box=False)