import os
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import numpy as np
import torch

def get_sam_predictor(cfg) -> SamPredictor:
    if cfg.sam_variant == "sam":
        sam = sam_model_registry[cfg.sam_encoder_version](checkpoint=cfg.sam_checkpoint_path)
        sam.to(cfg.device)
        sam_predictor = SamPredictor(sam)
        return sam_predictor
    
    if cfg.sam_variant == "mobilesam":
        from MobileSAM.setup_mobile_sam import setup_model
        # MOBILE_SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/mobile_sam.pt")
        # checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
        checkpoint = torch.load(cfg.mobile_sam_path)
        mobile_sam = setup_model()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=cfg.device)
        
        sam_predictor = SamPredictor(mobile_sam)
        return sam_predictor

    elif cfg.sam_variant == "lighthqsam":
        from LightHQSAM.setup_light_hqsam import setup_model
        HQSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/sam_hq_vit_tiny.pth")
        checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
        light_hqsam = setup_model()
        light_hqsam.load_state_dict(checkpoint, strict=True)
        light_hqsam.to(device=cfg.device)
        
        sam_predictor = SamPredictor(light_hqsam)
        return sam_predictor
        
    elif cfg.sam_variant == "fastsam":
        raise NotImplementedError
    else:
        raise NotImplementedError

# Prompting SAM with detected boxes in a batch
def get_sam_segmentation_from_xyxy_batched(sam_predictor: SamPredictor, image: np.ndarray, xyxy_tensor: torch.Tensor) -> torch.Tensor:
    
    sam_predictor.set_image(image)
    
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(xyxy_tensor, image.shape[:2])
    
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    
    return masks.squeeze()

# Prompting SAM with detected boxes in a batch
def get_sam_segmentation_from_xyxy(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    
    sam_predictor.set_image(image)
    
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)
    
