
import gzip
from pathlib import Path
import pickle
import cv2
import hydra
from line_profiler import profile
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import trange
from ultralytics import YOLO
from ultralytics import SAM
import supervision as sv
import open_clip


from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import vis_result_fast, save_video_detections
from conceptgraph.utils.general_utils import get_det_out_path, get_exp_out_path, get_vis_out_path, measure_time, save_hydra_config, ObjectClasses
from conceptgraph.utils.model_utils import compute_clip_features_batched 


@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="streamlined_detections")
@profile
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
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to(cfg.device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    
    # Set the classes for the detection model
    
    # classes = [line.strip() for line in open(cfg.classes_file)]
    # classes = [cls for cls in classes if cls not in bg_classes] # remove background classes
    obj_classes = ObjectClasses(cfg.classes_file, bg_classes=cfg.bg_classes, skip_bg=cfg.skip_bg)
    detection_model.set_classes(obj_classes.get_classes_arr())
    
    # Create the output directory and save the current config
    exp_out_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.exp_suffix)
    # exp_out_path = Path(cfg.dataset_root) / cfg.scene_id / "exps" / f"exp_{cfg.exp_suffix}"
    vis_folder_path = get_vis_out_path(exp_out_path)
    detections_folder_path = get_det_out_path(exp_out_path)

    save_hydra_config(cfg, exp_out_path)
    
    # Loop through the dataset and perform detections
    for idx in trange(len(dataset)):
        
        # Relevant paths and load image
        color_path = dataset.color_paths[idx]
        vis_save_path = vis_folder_path / Path(color_path).name

        # opencv can't read Path objects...
        vis_save_path = str(vis_save_path)
        image = cv2.imread(color_path) # This will in BGR color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
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

        # Compute and save the clip features of detections  
        image_crops, image_feats, text_feats = compute_clip_features_batched(
            image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, obj_classes.get_classes_arr(), cfg.device)

        #Visualize and save the annotated image
        annotated_image, labels = vis_result_fast(image, detections, obj_classes.get_classes_arr())
        cv2.imwrite(vis_save_path, annotated_image)
        
        # Save results 
        # Convert the detections to a dict. The elements are in np.array
        results = {
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask,
            "classes": obj_classes.get_classes_arr(),
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats,
        }
        
        # save the detections using pickle
        # Here we use gzip to compress the file, 
        # which could reduce the file size by 500x
        detections_name = (Path(vis_save_path).stem + ".pkl.gz")
        with gzip.open(detections_folder_path / detections_name , "wb") as f:
            pickle.dump(results, f)
        
    if cfg.save_video:
        save_video_detections(exp_out_path)
        

if __name__ == "__main__":
    measure_time(main)()