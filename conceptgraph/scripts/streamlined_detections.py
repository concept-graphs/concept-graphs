
from pathlib import Path
from conceptgraph.scripts.generate_gsa_results import get_sam_predictor
import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import trange
from ultralytics import YOLO
from ultralytics import SAM
import supervision as sv

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import vis_result_fast, vis_result_slow_caption, save_video_detections
from conceptgraph.utils.general_utils import measure_time, save_hydra_config, cfg_to_dict, prjson
from conceptgraph.utils.model_utils import get_sam_predictor,  get_sam_segmentation_from_xyxy_batched, get_sam_segmentation_from_xyxy


@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="streamlined_detections")
def main(cfg: DictConfig):

    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.desired_height,
        desired_width=cfg.desired_width,
        device="cpu",
        dtype=torch.float,
    )

    # Initialize the model(s)
    detection_model = measure_time(YOLO)('yolov8l-world.pt')
    sam_predictor = SAM('mobile_sam.pt') # UltraLytics SAM
    # sam_predictor = measure_time(get_sam_predictor)(cfg) # Normal SAM
    
    # Set the classes for the detection model
    bg_classes = ["wall", "floor", "ceiling"]
    classes = [line.strip() for line in open(cfg.classes_file)]
    classes = [cls for cls in classes if cls not in bg_classes] # remove background classes
    detection_model.set_classes(classes)
    
    # Create the output directory and save the current config
    exp_out_path = Path(cfg.dataset_root) / cfg.scene_id / f"exp_{cfg.exp_suffix}"
    (exp_out_path / "vis").mkdir(exist_ok=True, parents=True)
    save_hydra_config(cfg, exp_out_path)
    
    # Loop through the dataset and perform detections
    for idx in trange(len(dataset)):
        
        # Relevant paths and load image
        color_path = dataset.color_paths[idx]
        vis_save_path = exp_out_path / f"vis" / Path(color_path).name
        # opencv can't read Path objects...
        vis_save_path = str(vis_save_path)
        image = cv2.imread(color_path) # This will in BGR color space
            
        # Do initial object detection
        results = detection_model.predict(color_path, conf=0.1, verbose=False)
        confidences = results[0].boxes.conf.cpu().numpy()
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()
        
        # Get Masks Using SAM or MobileSAM
        # UltraLytics SAM
        sam_out = sam_predictor.predict(color_path, bboxes=xyxy_tensor, verbose=False)
        masks_tensor = sam_out[0].masks.data
        
        # Normal SAM
        # masks_tensor = get_sam_segmentation_from_xyxy_batched(
        #     sam_predictor=sam_predictor,
        #     image=image_rgb,
        #     xyxy_tensor=xyxy_tensor,
        # )
        
        masks_np = masks_tensor.cpu().numpy()
        
        # Create a detections object that we will save later
        detections = sv.Detections(
            xyxy=xyxy_np,
            confidence=confidences,
            class_id=detection_class_ids,
            mask=masks_np,
        )

        #Visualize the results
        annotated_image, labels = vis_result_fast(image, detections, classes)
        cv2.imwrite(vis_save_path, annotated_image)
        
        
    if cfg.save_video:
        save_video_detections(exp_out_path)
        

if __name__ == "__main__":
    measure_time(main)()