import os
from conceptgraph.utils.general_utils import measure_time
# from line_profiler import profile
# from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cosine

# def get_sam_predictor(cfg) -> SamPredictor:
#     if cfg.sam_variant == "sam":
#         sam = sam_model_registry[cfg.sam_encoder_version](checkpoint=cfg.sam_checkpoint_path)
#         sam.to(cfg.device)
#         sam_predictor = SamPredictor(sam)
#         return sam_predictor
    
#     if cfg.sam_variant == "mobilesam":
#         from MobileSAM.setup_mobile_sam import setup_model
#         # MOBILE_SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/mobile_sam.pt")
#         # checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
#         checkpoint = torch.load(cfg.mobile_sam_path)
#         mobile_sam = setup_model()
#         mobile_sam.load_state_dict(checkpoint, strict=True)
#         mobile_sam.to(device=cfg.device)
        
#         sam_predictor = SamPredictor(mobile_sam)
#         return sam_predictor

#     elif cfg.sam_variant == "lighthqsam":
#         from LightHQSAM.setup_light_hqsam import setup_model
#         HQSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/sam_hq_vit_tiny.pth")
#         checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
#         light_hqsam = setup_model()
#         light_hqsam.load_state_dict(checkpoint, strict=True)
#         light_hqsam.to(device=cfg.device)
        
#         sam_predictor = SamPredictor(light_hqsam)
#         return sam_predictor
        
#     elif cfg.sam_variant == "fastsam":
#         raise NotImplementedError
#     else:
#         raise NotImplementedError

# Prompting SAM with detected boxes in a batch
def get_sam_segmentation_from_xyxy_batched(sam_predictor, image: np.ndarray, xyxy_tensor: torch.Tensor) -> torch.Tensor:
    
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
def get_sam_segmentation_from_xyxy(sam_predictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    
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
    
def compute_clip_features(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
    backup_image = image.copy()
    
    image = Image.fromarray(image)
    
    # padding = args.clip_padding  # Adjust the padding amount as needed
    padding = 20  # Adjust the padding amount as needed
    
    image_crops = []
    image_feats = []
    text_feats = []

    
    for idx in range(len(detections.xyxy)):
        # Get the crop of the mask with padding
        x_min, y_min, x_max, y_max = detections.xyxy[idx]

        # Check and adjust padding to avoid going beyond the image borders
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        # Apply the adjusted padding
        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        # Get the preprocessed image for clip from the crop 
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0).to("cuda")

        crop_feat = clip_model.encode_image(preprocessed_image)
        crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
        
        class_id = detections.class_id[idx]
        tokenized_text = clip_tokenizer([classes[class_id]]).to("cuda")
        text_feat = clip_model.encode_text(tokenized_text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        
        crop_feat = crop_feat.cpu().numpy()
        text_feat = text_feat.cpu().numpy()

        image_crops.append(cropped_image)
        image_feats.append(crop_feat)
        text_feats.append(text_feat)
        
    # turn the list of feats into np matrices
    image_feats = np.concatenate(image_feats, axis=0)
    text_feats = np.concatenate(text_feats, axis=0)

    return image_crops, image_feats, text_feats

# @profile
def compute_clip_features_batched(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):
    
    image = Image.fromarray(image)
    padding = 20  # Adjust the padding amount as needed
    
    image_crops = []
    preprocessed_images = []
    text_tokens = []
    
    # Prepare data for batch processing
    for idx in range(len(detections.xyxy)):
        x_min, y_min, x_max, y_max = detections.xyxy[idx]
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
        preprocessed_images.append(preprocessed_image)

        class_id = detections.class_id[idx]
        text_tokens.append(classes[class_id])
        image_crops.append(cropped_image)
    
    # Convert lists to batches
    preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)
    text_tokens_batch = clip_tokenizer(text_tokens).to(device)
    
    # Batch inference
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocessed_images_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        text_features = clip_model.encode_text(text_tokens_batch)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Convert to numpy
    image_feats = image_features.cpu().numpy()
    text_feats = text_features.cpu().numpy()
    # image_feats = []
    # text_feats = []
    
    return image_crops, image_feats, text_feats


def compute_ft_vector_closeness_statistics(unbatched, batched):
    # Initialize lists to store statistics
    mad = []  # Mean Absolute Difference
    max_diff = []  # Maximum Absolute Difference
    mrd = []  # Mean Relative Difference
    cosine_sim = []  # Cosine Similarity

    for i in range(len(unbatched)):
        diff = np.abs(unbatched[i] - batched[i])
        mad.append(np.mean(diff))
        max_diff.append(np.max(diff))
        mrd.append(np.mean(diff / (np.abs(batched[i]) + 1e-8)))  # Adding a small value to avoid division by zero
        cosine_sim.append(1 - cosine(unbatched[i].flatten(), batched[i].flatten()))  # 1 - cosine distance to get similarity

    # Convert lists to numpy arrays for easy statistics
    mad = np.array(mad)
    max_diff = np.array(max_diff)
    mrd = np.array(mrd)
    cosine_sim = np.array(cosine_sim)

    # Print statistics
    print(f"Mean Absolute Difference: {np.mean(mad)}")
    print(f"Maximum Absolute Difference: {np.max(max_diff)}")
    print(f"Mean Relative Difference: {np.mean(mrd)}")
    print(f"Mean Cosine Similarity: {np.mean(cosine_sim)}")
    print(f"Min Cosine Similarity: {np.min(cosine_sim)}")