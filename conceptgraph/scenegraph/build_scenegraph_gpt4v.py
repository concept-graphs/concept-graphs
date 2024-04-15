from dataclasses import dataclass
import os
from pathlib import Path
import gzip
import pickle
import io
import base64
import json
import traceback

# Related third party imports
import tyro
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import imageio
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

import openai

from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.scripts.visualize_cfslam_results import load_result

def get_det_result(color_path, detection_folder_name="gsa_detections_ram_withbg_allclasses"):
    color_path = Path(color_path)
    detections_path = color_path.parent.parent / detection_folder_name / color_path.name
    detections_path = detections_path.with_suffix(".pkl.gz")
    color_path = str(color_path)
    detections_path = str(detections_path)
    
    with gzip.open(detections_path, "rb") as f:
        gobs = pickle.load(f)
    return gobs


def compute_norm_dist(obj_a, obj_b):
    '''
    Compute the distance between two objects, normalized by their extents
    '''
    points_a = np.asarray(obj_a['pcd'].points)
    center_a = points_a.mean(axis=0)
    extent_a = points_a.max(axis=0) - points_a.min(axis=0)
    
    points_b = np.asarray(obj_b['pcd'].points)
    center_b = points_b.mean(axis=0)
    extent_b = points_b.max(axis=0) - points_b.min(axis=0)
    
    dist = np.absolute(center_a - center_b)
    norm_dist = dist / ((extent_a + extent_b) / 2.0)
    
    return norm_dist

    
def compute_mutual_vis_ratio(obj_a, obj_b):
    '''
    Compute the product the visibility ratio of a pair of objects. 
    
    Visibility ratio is the ratio between the number of points visible in one view 
    and that of the fully built 3D object. 
    
    In order for GPT to reason about the relationship between two objects, we want 
    to get some images where both objects are mostly visible. So we compute this ratio
    and use it to select relevant images. 
    '''
    n_images = max(
        np.asarray(obj_a['image_idx']).max(),
        np.asarray(obj_b['image_idx']).max()
    ) + 1
    
    n_points_a = len(obj_a['pcd'].points)
    n_points_b = len(obj_b['pcd'].points)
    
    vis_ratio_a = np.zeros(n_images)
    vis_ratio_b = np.zeros(n_images)

    image_idx_a = np.asarray(obj_a['image_idx'])
    image_idx_b = np.asarray(obj_b['image_idx'])
    
    vis_points_a = np.asarray(obj_a['n_points'])
    vis_points_b = np.asarray(obj_b['n_points'])
    
    vis_ratio_a[image_idx_a] = vis_points_a / n_points_a
    vis_ratio_b[image_idx_b] = vis_points_b / n_points_b
    
    mutual_vis_ratio = vis_ratio_a * vis_ratio_b
    
    return mutual_vis_ratio, vis_ratio_a, vis_ratio_b


def resize_mask(mask, h, w):
    mask = mask.astype(float)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5)
    return mask


def draw_mask_on_image(mask, image, color, contour_width=3):
    '''
    Draw the mask on image. The contour of the mask is first computed and draw on the 
    image in dark color (e.g. dark red). Then the inner region of the mask is draw on 
    image in light color (e.g. light red). The original content of the image in the
    masked region will still be visible
    
    mask: numpy.ndarray of shape (H, W), dtype bool
    image: numpy.ndarray of shape (H, W, 3), dtype uint8
    '''
    # Copy the image to avoid altering the original
    output_image = image.copy()

    # Create darker and lighter shades of the provided color
    darker_color = tuple([x // 2 for x in color])  # Darken by reducing brightness by half
    lighter_color = tuple([min(x + 100, 255) for x in color])  # Lighten by increasing brightness
    
    # Create a colored version of the mask
    colored_mask = image # preserve the original image in the blending process
    colored_mask[mask] = lighter_color
    
    # Draw the filled mask with the lighter color, use a blend to keep underlying details
    cv2.addWeighted(output_image, 0.8, colored_mask, 0.2, 0, output_image)

    # Find contours of the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the contours with the darker color
    cv2.drawContours(output_image, contours, -1, darker_color, contour_width)

    return output_image


def draw_number_in_mask(image, binary_mask, text, color, font_scale = 1, thickness = 2):
    """
    Find proper places to draw text given a binary mask, and then draw the text on the image
    """
    binary_mask = binary_mask.astype(np.uint8)
    binary_mask = np.pad(binary_mask, ((1, 1), (1, 1)), 'constant')
    mask_dt = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 0)
    mask_dt = mask_dt[1:-1, 1:-1]
    max_dist = np.max(mask_dt)
    coords_y, coords_x = np.where(mask_dt == max_dist)  # coords is [y, x]
     
    x = coords_x[len(coords_x)//2]
    y = coords_y[len(coords_y)//2]
    
    # Determine font for the text.
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Calculate the size of the text to adjust the coordinates.
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Modify x, y coordinates to center the text by adjusting with the text size.
    x -= text_size[0] // 2  # Center horizontally
    y += text_size[1] // 2  # Center vertically

    # Draw the text at the chosen position with given color and font settings.
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

    return image


def crop_image_with_padding(image, mask, padding):
    """
    Crops an image using a boolean mask with added padding.

    Parameters:
        image (np.array): The input image as a numpy array of shape (H, W, 3).
        mask (np.array): A boolean mask of shape (H, W) where True indicates the pixels to keep.
        padding (int): The number of pixels to add as padding around the crop.

    Returns:
        np.array: The cropped and padded image.
    """
    if image.shape[:2] != mask.shape:
        raise ValueError("The dimensions of the image and the mask must match")
    
    # Finding the bounding box from the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Adding padding to the coordinates
    rmin = max(rmin - padding, 0)
    rmax = min(rmax + padding, image.shape[0])
    cmin = max(cmin - padding, 0)
    cmax = min(cmax + padding, image.shape[1])

    # Cropping the image
    cropped_image = image[rmin:rmax+1, cmin:cmax+1]

    return cropped_image


def encode_numpy_image_to_base64(np_image: np.ndarray) -> str:
    """
    Encodes a numpy array (HxWx3, dtype uint8) representing an image into a base64 string.
    
    Args:
    np_image (np.ndarray): A HxWx3 numpy array with dtype uint8, representing an RGB image.

    Returns:
    str: The base64 encoded string of the image.
    """
    # Check if input is a valid HxWx3 numpy array of dtype uint8
    if not (isinstance(np_image, np.ndarray) and np_image.ndim == 3 and np_image.shape[2] == 3 and np_image.dtype == np.uint8):
        raise ValueError("Input must be a HxWx3 numpy array with dtype uint8.")
    
    # Convert numpy array to PIL Image
    image = Image.fromarray(np_image)
    
    # Save the PIL Image to a bytes buffer
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='JPEG')  # Can use PNG or other formats depending on the requirement
    
    # Get bytes data from the buffer and encode it in base64
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    
    return base64_str


SYSTEM_PROMPT = '''
    You are an agent specialized in describing the spatial relationships between objects in an annotated image.
    
    You will be provided with an annotated image, where two objects are highlighted. The object highlighted in red is object A and the object highlighted in green is object B respectively. You will analyze the image and decide whether there is a spatial relationship between them. If there is, you will select one relationship from the following list: 
    
    - 0 - No relationship: return this if the pair of objects does not fall in any other relationships. 
    - 1 - A on top of B: Object A is on top of object B. Return this relationship if the object A is physically supported by the object B in the 3D space. Do NOT return this if A is just visually above B in 2D or if A is placed on the top of B but there is no supporting relationship. 
    - 2 - B on top of A: Object B is on top of object A. Return this relationship if the object B is physically supported by the object A in the 3D space. Do NOT return this if B is just visually above A in 2D or if B is placed on the top of A but there is no supporting relationship.
    
    Return the result in JSON in the following format:
    
    {
        "type": "<A number index of the relationship, e.g. 0>",
        "relationship": "<Relationship in text, e.g. A on top of B>",
        "reason": "<Explain briefly what object A and B are and your reasoning about the spatial relationship between A and B>",
    }
'''

def parse_relation(image: np.ndarray, client: openai.OpenAI):
    base64_image = encode_numpy_image_to_base64(image)
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-2024-04-09",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Return the spatial relationships between the highlighted objects in the image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            },
        ],
        max_tokens=512,
        top_p=0.1
    )

    return response.choices[0].message.content

@dataclass
class Main:
    replica_root = "/rvl-home/guqiao/rdata/Replica/"
    dataset_config_path = "./dataset/dataconfigs/replica/replica.yaml"
    
    scene_name = "room0"
    result_path = "/rvl-home/guqiao/rdata/Replica/room0/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_masksub_20230814_164846_post.pkl.gz"
    detection_folder_name="gsa_detections_ram_withbg_allclasses"
    save_path = "/rvl-home/guqiao/rdata/Replica/room0/sg_gpt4v/relations.json"
    
    n_api_retry: int = 5

    def main(self) -> None:
        assert os.getenv('OPENAI_API_KEY') is not None, "Please set the OPENAI_API_KEY environment variable"
        
        client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        
        dataset = get_dataset(
            dataconfig = self.dataset_config_path, 
            stride=5,
            basedir=self.replica_root,
            sequence=self.scene_name,
            device="cpu",
            dtype=torch.float,
        )
        
        objects, bg_objects, class_colors = load_result(self.result_path)
        
        print(f"Loaded {len(objects)} objects and {len(bg_objects)} bg objects")
        
        relations = []
        
        # Iterate over all pairs of objects
        n_objects = len(objects)
        for i in range(1, n_objects):
            for j in range(i+1, n_objects):
                i, j = np.random.choice(n_objects, 2, replace=False)

                obj_a = objects[i]
                obj_b = objects[j]
                
                # filter object pairs by their geometric distance
                norm_dist = compute_norm_dist(obj_a, obj_b)
                
                if norm_dist.max() > 1.5:
                    # Skip this pair as they are too far away
                    continue
                
                # Get the image with maximum mutual visiblity ratio
                mutual_vis_ratio, vis_ratio_a, vis_ratio_b = compute_mutual_vis_ratio(obj_a, obj_b)
                image_id, max_mutual_vis_ratio = mutual_vis_ratio.argmax(), mutual_vis_ratio.max()
                
                if max_mutual_vis_ratio < 0.1:
                    # Skip if they are not co-visible at any image
                    continue
                
                # print(image_id, max_mutual_vis_ratio)
                color, _, _, _ = dataset[image_id]
                color = color.detach().cpu().numpy()
                color = color.round().astype(np.uint8)
                
                det_id_a = np.where(np.asarray(obj_a['image_idx']) == image_id)[0][0]
                mask_a = obj_a['mask'][det_id_a]
                mask_a = resize_mask(mask_a, color.shape[0], color.shape[1])
                
                det_id_b = np.where(np.asarray(obj_b['image_idx']) == image_id)[0][0]
                mask_b = obj_b['mask'][det_id_b]
                mask_b = resize_mask(mask_b, color.shape[0], color.shape[1])
                
                blend_image = draw_mask_on_image(mask_a, color, (255, 0, 0), contour_width=3)
                blend_image = draw_mask_on_image(mask_b, blend_image, (0, 255, 0), contour_width=3)
                
                blend_image = draw_number_in_mask(blend_image, mask_a, "A", (255, 255, 255), font_scale=1, thickness=2)
                blend_image = draw_number_in_mask(blend_image, mask_b, "B", (255, 255, 255), font_scale=1, thickness=2)
                
                # mask_union = np.logical_or(mask_a, mask_b)
                # blend_image = crop_image_with_padding(blend_image, mask_union, padding=50)
                
                # plt.imshow(blend_image)
                # plt.axis("off")
                # plt.show()
                
                api_success = False
                for _ in range(self.n_api_retry):
                    try:
                        relation_res = parse_relation(blend_image)
                        relation = json.loads(relation_res)
                    except Exception as e:
                        print(traceback.format_exc())
                        print("Failed to get relation, retrying...")
                        continue

                    api_success = True
                    break
                    
                if not api_success:
                    relation = {
                        "type": 0,
                        "relationship": "No relationship",
                        "reason": "Failed to get relation from the API"
                    }
                    
                if type(relation['type']) == str:
                    relation['type'] = int(relation['type'])
                    
                relation['idx_a'] = i
                relation['idx_b'] = j
                relation['image_id'] = image_id
                
                print(relation)

                relations.append(relation)
                
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))
            
        with open(self.save_path, "w") as f:
            json.dump(relations, f)
            
        print(f"Saved {len(relations)} relations to {self.save_path}")

if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Main).main()