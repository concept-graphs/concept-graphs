"""
Build a scene graph from the segment-based map and captions from LLaVA.
"""

import gc
import gzip
import json
import os
import pickle as pkl
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Literal, Union
from textwrap import wrap

from conceptgraph.utils.general_utils import prjson

import cv2
import matplotlib.pyplot as plt

import numpy as np
import rich
import torch
import tyro
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from tqdm import tqdm, trange
from transformers import logging as hf_logging

# from mappingutils import (
#     MapObjectList,
#     compute_3d_giou_accuracte_batch,
#     compute_3d_iou_accuracte_batch,
#     compute_iou_batch,
#     compute_overlap_matrix_faiss,
#     num_points_closer_than_threshold_batch,
# )

torch.autograd.set_grad_enabled(False)
hf_logging.set_verbosity_error()

# Import OpenAI API
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class ProgramArgs:
    mode: Literal[
        "extract-node-captions",
        "refine-node-captions",
        "build-scenegraph",
        "generate-scenegraph-json",
        "annotate-scenegraph",
    ]

    # Path to cache directory
    cachedir: str = "saved/room0"
    
    prompts_path: str = "prompts/gpt_prompts.json"

    # Path to map file
    mapfile: str = "saved/room0/map/scene_map_cfslam.pkl.gz"

    # Path to file storing segment class names
    class_names_file: str = "saved/room0/gsa_classes_ram.json"

    # Device to use
    device: str = "cuda:0"

    # Voxel size for downsampling
    downsample_voxel_size: float = 0.025

    # Maximum number of detections to consider, per object
    max_detections_per_object: int = 10

    # Suppress objects with less than this number of observations
    min_views_per_object: int = 2

    # List of objects to annotate (default: all objects)
    annot_inds: Union[List[int], None] = None

    # Masking option
    masking_option: Literal["blackout", "red_outline", "none"] = "none"

def load_scene_map(args, scene_map):
    """
    Loads a scene map from a gzip-compressed pickle file. This is a function because depending whether the mapfile was made using cfslam_pipeline_batch.py or merge_duplicate_objects.py, the file format is different (see below). So this function handles that case.
    
    The function checks the structure of the deserialized object to determine
    the correct way to load it into the `scene_map` object. There are two
    expected formats:
    1. A dictionary containing an "objects" key.
    2. A list or a dictionary (replace with your expected type).
    """
    
    with gzip.open(Path(args.mapfile), "rb") as f:
        loaded_data = pkl.load(f)
        
        # Check the type of the loaded data to decide how to proceed
        if isinstance(loaded_data, dict) and "objects" in loaded_data:
            scene_map.load_serializable(loaded_data["objects"])
        elif isinstance(loaded_data, list) or isinstance(loaded_data, dict):  # Replace with your expected type
            scene_map.load_serializable(loaded_data)
        else:
            raise ValueError("Unexpected data format in map file.")
        print(f"Loaded {len(scene_map)} objects")



def crop_image_pil(image: Image, x1: int, y1: int, x2: int, y2: int, padding: int = 0) -> Image:
    """
    Crop the image with some padding

    Args:
        image: PIL image
        x1, y1, x2, y2: bounding box coordinates
        padding: padding around the bounding box

    Returns:
        image_crop: PIL image

    Implementation from the CFSLAM repo
    """
    image_width, image_height = image.size
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding)

    image_crop = image.crop((x1, y1, x2, y2))
    return image_crop


def draw_red_outline(image, mask):
    """ Draw a red outline around the object i nan image"""
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    red_outline = [255, 0, 0]

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red outlines around the object. The last argument "3" indicates the thickness of the outline.
    cv2.drawContours(image_np, contours, -1, red_outline, 3)

    # Optionally, add padding around the object by dilating the drawn contours
    kernel = np.ones((5, 5), np.uint8)
    image_np = cv2.dilate(image_np, kernel, iterations=1)
    
    image_pil = Image.fromarray(image_np)

    return image_pil


def crop_image_and_mask(image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
    """ Crop the image and mask with some padding. I made a single function that crops both the image and the mask at the same time because I was getting shape mismatches when I cropped them separately.This way I can check that they are the same shape."""
    
    image = np.array(image)
    # Verify initial dimensions
    if image.shape[:2] != mask.shape:
        print("Initial shape mismatch: Image shape {} != Mask shape {}".format(image.shape, mask.shape))
        return None, None

    # Define the cropping coordinates
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    # round the coordinates to integers
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

    # Crop the image and the mask
    image_crop = image[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    # Verify cropped dimensions
    if image_crop.shape[:2] != mask_crop.shape:
        print("Cropped shape mismatch: Image crop shape {} != Mask crop shape {}".format(image_crop.shape, mask_crop.shape))
        return None, None
    
    # convert the image back to a pil image
    image_crop = Image.fromarray(image_crop)

    return image_crop, mask_crop

def blackout_nonmasked_area(image_pil, mask):
    """ Blackout the non-masked area of an image"""
    # convert image to numpy array
    image_np = np.array(image_pil)
    # Create an all-black image of the same shape as the input image
    black_image = np.zeros_like(image_np)
    # Wherever the mask is True, replace the black image pixel with the original image pixel
    black_image[mask] = image_np[mask]
    # convert back to pil image
    black_image = Image.fromarray(black_image)
    return black_image

def plot_images_with_captions(images, captions, confidences, low_confidences, masks, savedir, idx_obj):
    """ This is debug helper function that plots the images with the captions and masks overlaid and saves them to a directory. This way you can inspect exactly what the LLaVA model is captioning which image with the mask, and the mask confidence scores overlaid."""
    
    n = min(9, len(images))  # Only plot up to 9 images
    nrows = int(np.ceil(n / 3))
    ncols = 3 if n > 1 else 1
    fig, axarr = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), squeeze=False)  # Adjusted figsize

    for i in range(n):
        row, col = divmod(i, 3)
        ax = axarr[row][col]
        ax.imshow(images[i])

        # Apply the mask to the image
        img_array = np.array(images[i])
        if img_array.shape[:2] != masks[i].shape:
            ax.text(0.5, 0.5, "Plotting error: Shape mismatch between image and mask", ha='center', va='center')
        else:
            green_mask = np.zeros((*masks[i].shape, 3), dtype=np.uint8)
            green_mask[masks[i]] = [0, 255, 0]  # Green color where mask is True
            ax.imshow(green_mask, alpha=0.15)  # Overlay with transparency

        title_text = f"Caption: {captions[i]}\nConfidence: {confidences[i]:.2f}"
        if low_confidences[i]:
            title_text += "\nLow Confidence"
        
        # Wrap the caption text
        wrapped_title = '\n'.join(wrap(title_text, 30))
        
        ax.set_title(wrapped_title, fontsize=12)  # Reduced font size for better fitting
        ax.axis('off')

    # Remove any unused subplots
    for i in range(n, nrows * ncols):
        row, col = divmod(i, 3)
        axarr[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(savedir / f"{idx_obj}.png")
    plt.close()

def save_image_list(idx_obj, cache_path, image_list):
    savedir_images_obj = cache_path / f"{idx_obj}"
    savedir_images_obj.mkdir(exist_ok=True, parents=True)
    for idx, image in enumerate(image_list):
        image.save(savedir_images_obj / f"{idx}.png")


def extract_node_captions(args):
    from conceptgraph.llava.llava_model import LLaVaChat

    # NOTE: args.mapfile is in cfslam format
    from conceptgraph.slam.slam_classes import MapObjectList

    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)
    
    # Scene map is in CFSLAM format
    # keys: 'image_idx', 'mask_idx', 'color_path', 'class_id', 'num_detections',
    # 'mask', 'xyxy', 'conf', 'n_points', 'pixel_area', 'contain_number', 'clip_ft',
    # 'text_ft', 'pcd_np', 'bbox_np', 'pcd_color_np'

    # Imports to help with feature extraction
    # from extract_mask_level_features import (
    #     crop_bbox_from_img,
    #     get_model_and_preprocessor,
    #     preprocess_and_encode_pil_image,
    # )

    # Load class names from the json file
    class_names = None
    with open(Path(args.class_names_file), "r") as f:
        class_names = [cls.strip() for cls in f.readlines()]
    print(class_names)

    # Creating a namespace object to pass args to the LLaVA chat object
    chat_args = SimpleNamespace()
    chat_args.model_path = os.getenv("LLAVA_MODEL_PATH")
    chat_args.conv_mode = "v0_mmtag" # "multimodal"
    chat_args.num_gpus = 1

    # rich console for pretty printing
    console = rich.console.Console()

    # Initialize LLaVA chat
    chat = LLaVaChat(chat_args.model_path, chat_args.conv_mode, chat_args.num_gpus)
    # chat = LLaVaChat(chat_args)
    print("LLaVA chat initialized...")
    query = "Describe the central object in the image."
    # query = "Describe the object in the image that is outlined in red."

    # Directories to save features and captions
    savedir_feat = Path(args.cachedir) / "cfslam_feat_llava"
    savedir_feat.mkdir(exist_ok=True, parents=True)
    savedir_captions = Path(args.cachedir) / "cfslam_captions_llava"
    savedir_captions.mkdir(exist_ok=True, parents=True)
    savedir_debug = Path(args.cachedir) / "cfslam_captions_llava_debug"
    savedir_debug.mkdir(exist_ok=True, parents=True)
    savedir_images = Path(args.cachedir) / "images" / "raw"
    savedir_images.mkdir(exist_ok=True, parents=True)
    savedir_images_modified = Path(args.cachedir) / "images" / "processed"
    savedir_images_modified.mkdir(exist_ok=True, parents=True)

    caption_dict_list = []

    for idx_obj, obj in tqdm(enumerate(scene_map), total=len(scene_map)):
        conf = obj["conf"]
        conf = np.array(conf)
        idx_most_conf = np.argsort(conf)[::-1]

        features = []
        captions = []
        low_confidences = []
        
        image_list = []
        image_modified_list = []
        caption_list = []
        confidences_list = []
        low_confidences_list = []
        mask_list = []  # New list for masks
        if len(idx_most_conf) < 2:
            continue 
        idx_most_conf = idx_most_conf[:args.max_detections_per_object]

        for idx_det in tqdm(idx_most_conf):
            # image = Image.open(correct_path).convert("RGB")
            image = Image.open(obj["color_path"][idx_det]).convert("RGB")
            xyxy = obj["xyxy"][idx_det]
            class_id = obj["class_id"][idx_det]
            class_name = class_names[class_id]
            # Retrieve and crop mask
            mask = obj["mask"][idx_det]

            padding = 10
            x1, y1, x2, y2 = xyxy
            # image_crop = crop_image_pil(image, x1, y1, x2, y2, padding=padding)
            image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
            if args.masking_option == "blackout":
                image_crop_modified = blackout_nonmasked_area(image_crop, mask_crop)
            elif args.masking_option == "red_outline":
                image_crop_modified = draw_red_outline(image_crop, mask_crop)
            else:
                image_crop_modified = image_crop  # No modification

            _w, _h = image_crop.size
            if _w * _h < 70 * 70:
                # captions.append("small object")
                print("small object. Skipping LLaVA captioning...")
                low_confidences.append(True)
                continue
            else:
                low_confidences.append(False)

            # image_tensor = chat.image_processor.preprocess(image_crop, return_tensors="pt")["pixel_values"][0]
            image_tensor = chat.image_processor.preprocess(image_crop_modified, return_tensors="pt")["pixel_values"][0]

            image_features = chat.encode_image(image_tensor[None, ...].half().cuda())
            features.append(image_features.detach().cpu())

            chat.reset()
            console.print("[bold red]User:[/bold red] " + query)
            outputs = chat(query=query, image_features=image_features)
            console.print("[bold green]LLaVA:[/bold green] " + outputs)
            captions.append(outputs)
        
            # print(f"Line 274, obj['mask'][idx_det].shape: {obj['mask'][idx_det].shape}")
            # print(f"Line 276, image.size: {image.size}")
            
            # For the LLava debug folder
            conf_value = conf[idx_det]
            image_list.append(image_crop)
            image_modified_list.append(image_crop_modified)
            caption_list.append(outputs)
            confidences_list.append(conf_value)
            low_confidences_list.append(low_confidences[-1])
            mask_list.append(mask_crop)  # Add the cropped mask

        caption_dict_list.append(
            {
                "id": idx_obj,
                "captions": captions,
                "low_confidences": low_confidences,
            }
        )

        # Concatenate the features
        if len(features) > 0:
            features = torch.cat(features, dim=0)

        # Save the feature descriptors
        torch.save(features, savedir_feat / f"{idx_obj}.pt")
        
        # Again for the LLava debug folder
        if len(image_list) > 0:
            plot_images_with_captions(image_list, caption_list, confidences_list, low_confidences_list, mask_list, savedir_debug, idx_obj)

        # Save images
        save_image_list(idx_obj, savedir_images, image_list)
        save_image_list(idx_obj, savedir_images_modified, image_modified_list)


    # Save the captions
    # Remove the "The central object in the image is " prefix from 
    # the captions as it doesnt convey and actual info
    for item in caption_dict_list:
        item["captions"] = [caption.replace("The central object in the image is ", "") for caption in item["captions"]]
    # Save the captions to a json file
    with open(Path(args.cachedir) / "cfslam_llava_captions.json", "w", encoding="utf-8") as f:
        json.dump(caption_dict_list, f, indent=4, sort_keys=False)


def save_json_to_file(json_str, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_str, f, indent=4, sort_keys=False)


def refine_node_captions(args):
    # NOTE: args.mapfile is in cfslam format
    from conceptgraph.slam.slam_classes import MapObjectList
    from conceptgraph.scenegraph.GPTPrompt import GPTPrompt

    # Load the captions for each segment
    caption_file = Path(args.cachedir) / "cfslam_llava_captions.json"
    captions = None
    with open(caption_file, "r") as f:
        captions = json.load(f)
    # loaddir_captions = Path(args.cachedir) / "cfslam_captions_llava"
    # captions = []
    # for idx_obj in range(len(os.listdir(loaddir_captions))):
    #     with open(loaddir_captions / f"{idx_obj}.pkl", "rb") as f:
    #         captions.append(pkl.load(f))

    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)
    
    # load the prompt
    gpt_messages = GPTPrompt().get_json()

    TIMEOUT = 25  # Timeout in seconds

    responses_savedir = Path(args.cachedir) / "cfslam_gpt-4_responses"
    responses_savedir.mkdir(exist_ok=True, parents=True)

    responses = []
    unsucessful_responses = 0

    # loop over every object
    for _i in trange(len(captions)):
        if len(captions[_i]) == 0:
            continue
        
        # Prepare the object prompt 
        _dict = {}
        _caption = captions[_i]
        _bbox = scene_map[_i]["bbox"]
        # _bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(scene_map[_i]["bbox"]))
        _dict["id"] = _caption["id"]
        # _dict["bbox_extent"] = np.round(_bbox.extent, 1).tolist()
        # _dict["bbox_center"] = np.round(_bbox.center, 1).tolist()
        _dict["captions"] = _caption["captions"]
        # _dict["low_confidences"] = _caption["low_confidences"]
        # Convert to printable string
        
        # Make and format the full prompt
        preds = json.dumps(_dict, indent=0)

        start_time = time.time()
    
        curr_chat_messages = gpt_messages[:]
        curr_chat_messages.append({"role": "user", "content": preds})
        chat_completion = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model="gpt-4-turbo-preview",
            messages=curr_chat_messages,
            timeout=TIMEOUT,  # Timeout in seconds
        )
        elapsed_time = time.time() - start_time
        if elapsed_time > TIMEOUT:
            print("Timed out exceeded!")
            _dict["response"] = "FAIL"
            # responses.append('{"object_tag": "FAIL"}')
            save_json_to_file(_dict, responses_savedir / f"{_caption['id']}.json")
            responses.append(json.dumps(_dict))
            unsucessful_responses += 1
            continue
        
        # count unsucessful responses
        if "invalid" in chat_completion["choices"][0]["message"]["content"].strip("\n"):
            unsucessful_responses += 1
            
        # print output
        prjson([{"role": "user", "content": preds}])
        print(chat_completion["choices"][0]["message"]["content"])
        print(f"Unsucessful responses so far: {unsucessful_responses}")
        _dict["response"] = chat_completion["choices"][0]["message"]["content"].strip("\n")
        
        # save the response
        responses.append(json.dumps(_dict))
        save_json_to_file(_dict, responses_savedir / f"{_caption['id']}.json")
        # responses.append(chat_completion["choices"][0]["message"]["content"].strip("\n"))

    # tags = []
    # for response in responses:
    #     try:
    #         parsed = json.loads(response)
    #         tags.append(parsed["object_tag"])
    #     except:
    #         tags.append("FAIL")

    # Save the responses to a text file
    # with open(Path(args.cachedir) / "gpt-3.5-turbo_responses.txt", "w") as f:
    #     for response in responses:
    #         f.write(response + "\n")
    with open(Path(args.cachedir) / "cfslam_gpt-4_responses.pkl", "wb") as f:
        pkl.dump(responses, f)


def extract_object_tag_from_json_str(json_str):
    start_str_found = False
    is_object_tag = False
    object_tag_complete = False
    object_tag = ""
    r = json_str.strip().split()
    for _idx, _r in enumerate(r):
        if not start_str_found:
            # Searching for open parenthesis of JSON
            if _r == "{":
                start_str_found = True
                continue
            else:
                continue
        # Start string found. Now skip everything until the object_tag field
        if not is_object_tag:
            if _r == '"object_tag":':
                is_object_tag = True
                continue
            else:
                continue
        # object_tag field found. Read it
        if is_object_tag and not object_tag_complete:
            if _r == '"':
                continue
            else:
                if _r.strip() in [",", "}"]:
                    break
                object_tag += f" {_r}"
                continue
    return object_tag


def build_scenegraph(args):
    from conceptgraph.slam.slam_classes import MapObjectList
    # from conceptgraph.slam.utils import compute_overlap_matrix
    from conceptgraph.slam.utils import compute_overlap_matrix_general

    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)

    response_dir = Path(args.cachedir) / "cfslam_gpt-4_responses"
    responses = []
    object_tags = []
    also_indices_to_remove = [] # indices to remove if the json file does not exist
    for idx in range(len(scene_map)):
        # check if the json file exists first 
        if not (response_dir / f"{idx}.json").exists():
            also_indices_to_remove.append(idx)
            continue
        with open(response_dir / f"{idx}.json", "r") as f:
            _d = json.load(f)
            try:
                _d["response"] = json.loads(_d["response"])
            except json.JSONDecodeError:
                _d["response"] = {
                    'summary': f'GPT4 json reply failed: Here is the invalid response {_d["response"]}',
                    'possible_tags': ['possible_tag_json_failed'],
                    'object_tag': 'invalid'
                }
            responses.append(_d)
            object_tags.append(_d["response"]["object_tag"])

    # # Load the responses from the json file
    # responses = None
    # # Load json file into a list
    # with open(Path(args.cachedir) / "cfslam_gpt-4_responses.pkl", "rb") as f:
    #     responses = pkl.load(f)
    # object_tags = []
    # for response in responses:
    #     # _tag = extract_object_tag_from_json_str(response)
    #     # _tag = _tag.lower().replace('"', "").strip()
    #     _d = json.loads(json.loads(response)["responses"])
    #     object_tags.append(_d["object_tag"])

    # Remove segments that correspond to "invalid" tags
    indices_to_remove = [i for i in range(len(responses)) if object_tags[i].lower() in ["fail", "invalid"]]
    # Also remove segments that do not have a minimum number of observations
    indices_to_remove = set(indices_to_remove)
    for obj_idx in range(len(scene_map)):
        conf = scene_map[obj_idx]["conf"]
        # Remove objects with less than args.min_views_per_object observations
        if len(conf) < args.min_views_per_object:
            indices_to_remove.add(obj_idx)
    indices_to_remove = list(indices_to_remove)
    # combine with also_indices_to_remove and sort the list
    indices_to_remove = list(set(indices_to_remove + also_indices_to_remove))
    
    # List of tags in original scene map that are in the pruned scene map
    segment_ids_to_retain = [i for i in range(len(scene_map)) if i not in indices_to_remove]
    with open(Path(args.cachedir) / "cfslam_scenegraph_invalid_indices.pkl", "wb") as f:
        pkl.dump(indices_to_remove, f)
    print(f"Removed {len(indices_to_remove)} segments")
    
    # Filtering responses based on segment_ids_to_retain
    responses = [resp for resp in responses if resp['id'] in segment_ids_to_retain]

    # Assuming each response dictionary contains an 'object_tag' key for the object tag.
    # Extract filtered object tags based on filtered_responses
    object_tags = [resp['response']['object_tag'] for resp in responses]


    pruned_scene_map = []
    pruned_object_tags = []
    for _idx, segmentidx in enumerate(segment_ids_to_retain):
        pruned_scene_map.append(scene_map[segmentidx])
        pruned_object_tags.append(object_tags[_idx])
    scene_map = MapObjectList(pruned_scene_map)
    object_tags = pruned_object_tags
    del pruned_scene_map
    # del pruned_object_tags
    gc.collect()
    num_segments = len(scene_map)

    for i in range(num_segments):
        scene_map[i]["caption_dict"] = responses[i]
        # scene_map[i]["object_tag"] = object_tags[i]

    # Save the pruned scene map (create the directory if needed)
    if not (Path(args.cachedir) / "map").exists():
        (Path(args.cachedir) / "map").mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "wb") as f:
        pkl.dump(scene_map.to_serializable(), f)

    print("Computing bounding box overlaps...")
    bbox_overlaps = compute_overlap_matrix_general(args, scene_map)

    # Construct a weighted adjacency matrix based on similarity scores
    weights = []
    rows = []
    cols = []
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            if i == j:
                continue
            if bbox_overlaps[i, j] > 0.01:
                weights.append(bbox_overlaps[i, j])
                rows.append(i)
                cols.append(j)
                weights.append(bbox_overlaps[i, j])
                rows.append(j)
                cols.append(i)

    adjacency_matrix = csr_matrix((weights, (rows, cols)), shape=(num_segments, num_segments))

    # Find the minimum spanning tree of the weighted adjacency matrix
    mst = minimum_spanning_tree(adjacency_matrix)

    # Find connected components in the minimum spanning tree
    _, labels = connected_components(mst)

    components = []
    _total = 0
    if len(labels) != 0:
        for label in range(labels.max() + 1):
            indices = np.where(labels == label)[0]
            _total += len(indices.tolist())
            components.append(indices.tolist())

    with open(Path(args.cachedir) / "cfslam_scenegraph_components.pkl", "wb") as f:
        pkl.dump(components, f)

    # Initialize a list to store the minimum spanning trees of connected components
    minimum_spanning_trees = []
    relations = []
    if len(labels) != 0:
        # Iterate over each connected component
        for label in range(labels.max() + 1):
            component_indices = np.where(labels == label)[0]
            # Extract the subgraph for the connected component
            subgraph = adjacency_matrix[component_indices][:, component_indices]
            # Find the minimum spanning tree of the connected component subgraph
            _mst = minimum_spanning_tree(subgraph)
            # Add the minimum spanning tree to the list
            minimum_spanning_trees.append(_mst)

        TIMEOUT = 25  # timeout in seconds

        if not (Path(args.cachedir) / "cfslam_object_relations.json").exists():
            relation_queries = []
            for componentidx, component in enumerate(components):
                if len(component) <= 1:
                    continue
                for u, v in zip(
                    minimum_spanning_trees[componentidx].nonzero()[0], minimum_spanning_trees[componentidx].nonzero()[1]
                ):
                    segmentidx1 = component[u]
                    segmentidx2 = component[v]
                    _bbox1 = scene_map[segmentidx1]["bbox"]
                    _bbox2 = scene_map[segmentidx2]["bbox"]

                    input_dict = {
                        "object1": {
                            "id": segmentidx1,
                            "bbox_extent": np.round(_bbox1.extent, 1).tolist(),
                            "bbox_center": np.round(_bbox1.center, 1).tolist(),
                            "object_tag": object_tags[segmentidx1],
                        },
                        "object2": {
                            "id": segmentidx2,
                            "bbox_extent": np.round(_bbox2.extent, 1).tolist(),
                            "bbox_center": np.round(_bbox2.center, 1).tolist(),
                            "object_tag": object_tags[segmentidx2],
                        },
                    }
                    print(f"{input_dict['object1']['object_tag']}, {input_dict['object2']['object_tag']}")

                    relation_queries.append(input_dict)

                    input_json_str = json.dumps(input_dict)

                    # Default prompt
                    DEFAULT_PROMPT = """
                    The input is a list of JSONs describing two objects "object1" and "object2". You need to produce a JSON
                    string (and nothing else), with two keys: "object_relation", and "reason".

                    Each of the JSON fields "object1" and "object2" will have the following fields:
                    1. bbox_extent: the 3D bounding box extents of the object
                    2. bbox_center: the 3D bounding box center of the object
                    3. object_tag: an extremely brief description of the object

                    Produce an "object_relation" field that best describes the relationship between the two objects. The
                    "object_relation" field must be one of the following (verbatim):
                    1. "a on b": if object a is an object commonly placed on top of object b
                    2. "b on a": if object b is an object commonly placed on top of object a
                    3. "a in b": if object a is an object commonly placed inside object b
                    4. "b in a": if object b is an object commonly placed inside object a
                    5. "none of these": if none of the above best describe the relationship between the two objects

                    Before producing the "object_relation" field, produce a "reason" field that explains why
                    the chosen "object_relation" field is the best.
                    """

                    start_time = time.time()
                    chat_completion = openai.ChatCompletion.create(
                        # model="gpt-3.5-turbo",
                        model="gpt-4",
                        messages=[{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + input_json_str}],
                        timeout=TIMEOUT,  # Timeout in seconds
                    )
                    elapsed_time = time.time() - start_time
                    output_dict = input_dict
                    if elapsed_time > TIMEOUT:
                        print("Timed out exceeded!")
                        output_dict["object_relation"] = "FAIL"
                        continue
                    else:
                        try:
                            # Attempt to parse the output as a JSON
                            chat_output_json = json.loads(chat_completion["choices"][0]["message"]["content"])
                            # If the output is a valid JSON, then add it to the output dictionary
                            output_dict["object_relation"] = chat_output_json["object_relation"]
                            output_dict["reason"] = chat_output_json["reason"]
                        except:
                            output_dict["object_relation"] = "FAIL"
                            output_dict["reason"] = "FAIL"
                    relations.append(output_dict)

                    # print(chat_completion["choices"][0]["message"]["content"])

            # Save the query JSON to file
            print("Saving query JSON to file...")
            with open(Path(args.cachedir) / "cfslam_object_relation_queries.json", "w") as f:
                json.dump(relation_queries, f, indent=4)

            # Saving the output
            print("Saving object relations to file...")
            with open(Path(args.cachedir) / "cfslam_object_relations.json", "w") as f:
                json.dump(relations, f, indent=4)
        else:
            relations = json.load(open(Path(args.cachedir) / "cfslam_object_relations.json", "r"))

    scenegraph_edges = []

    _idx = 0
    for componentidx, component in enumerate(components):
        if len(component) <= 1:
            continue
        for u, v in zip(
            minimum_spanning_trees[componentidx].nonzero()[0], minimum_spanning_trees[componentidx].nonzero()[1]
        ):
            segmentidx1 = component[u]
            segmentidx2 = component[v]
            # print(f"{segmentidx1}, {segmentidx2}, {relations[_idx]['object_relation']}")
            if relations[_idx]["object_relation"] != "none of these":
                scenegraph_edges.append((segmentidx1, segmentidx2, relations[_idx]["object_relation"]))
            _idx += 1
    print(f"Created 3D scenegraph with {num_segments} nodes and {len(scenegraph_edges)} edges")

    with open(Path(args.cachedir) / "cfslam_scenegraph_edges.pkl", "wb") as f:
        pkl.dump(scenegraph_edges, f)


def generate_scenegraph_json(args):
    from conceptgraph.slam.slam_classes import MapObjectList
    

    # Generate the JSON file summarizing the scene, if it doesn't exist already
    # or if the --recopmute_scenegraph_json flag is set
    scene_desc = []
    print("Generating scene graph JSON file...")

    # Load the pruned scene map
    scene_map = MapObjectList()
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "rb") as f:
        scene_map.load_serializable(pkl.load(f))
    print(f"Loaded scene map with {len(scene_map)} objects")

    for i, segment in enumerate(scene_map):
        _d = {
            "id": segment["caption_dict"]["id"],
            "bbox_extent": np.round(segment['bbox'].extent, 1).tolist(),
            "bbox_center": np.round(segment['bbox'].center, 1).tolist(),
            "possible_tags": segment["caption_dict"]["response"]["possible_tags"],
            "object_tag": segment["caption_dict"]["response"]["object_tag"],
            "caption": segment["caption_dict"]["response"]["summary"],
        }
        scene_desc.append(_d)
    with open(Path(args.cachedir) / "scene_graph.json", "w") as f:
        json.dump(scene_desc, f, indent=4)


def display_images(image_list):
    num_images = len(image_list)
    cols = 2  # Number of columns for the subplots (you can change this as needed)
    rows = (num_images + cols - 1) // cols

    _, axes = plt.subplots(rows, cols, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img = image_list[i]
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def annotate_scenegraph(args):
    from conceptgraph.slam.slam_classes import MapObjectList

    # Load the pruned scene map
    scene_map = MapObjectList()
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "rb") as f:
        scene_map.load_serializable(pkl.load(f))

    annot_inds = None
    if args.annot_inds is not None:
        annot_inds = args.annot_inds
    # If annot_inds is not None, we also need to load the annotation json file and only
    # annotate the objects that are specified in the annot_inds list
    annots = []
    if annot_inds is not None:
        annots = json.load(open(Path(args.cachedir) / "annotated_scenegraph.json", "r"))

    if annot_inds is None:
        annot_inds = list(range(len(scene_map)))

    for idx in annot_inds:
        print(f"Object {idx} out of {len(annot_inds)}...")
        obj = scene_map[idx]

        prev_annot = None
        if len(annots) >= idx + 1:
            prev_annot = annots[idx]

        annot = {}
        annot["id"] = idx

        conf = obj["conf"]
        conf = np.array(conf)
        idx_most_conf = np.argsort(conf)[::-1]
        print(obj.keys())

        imgs = []

        for idx_det in idx_most_conf:
            image = Image.open(obj["color_path"][idx_det]).convert("RGB")
            xyxy = obj["xyxy"][idx_det]
            mask = obj["mask"][idx_det]

            padding = 10
            x1, y1, x2, y2 = xyxy
            image_crop = crop_image_pil(image, x1, y1, x2, y2, padding=padding)
            mask_crop = crop_image_pil(Image.fromarray(mask), x1, y1, x2, y2, padding=padding)
            mask_crop = np.array(mask_crop)[..., None]
            mask_crop[mask_crop == 0] = 0.05
            image_crop = np.array(image_crop) * mask_crop
            imgs.append(image_crop)
            if len(imgs) >= 5:
                break
            # if idx_det >= 5:
            #     break

        # Display the images
        display_images(imgs)
        plt.close("all")

        # Ask the user to annotate the object
        if prev_annot is not None:
            print("Previous annotation:")
            print(prev_annot)
        annot["object_tags"] = input("Enter object tags (comma-separated): ")
        annot["colors"] = input("Enter colors (comma-separated): ")
        annot["materials"] = input("Enter materials (comma-separated): ")

        if prev_annot is not None:
            annots[idx] = annot
        else:
            annots.append(annot)

        go_on = input("Continue? (y/n): ")
        if go_on == "n":
            break

    # Save the annotations
    with open(Path(args.cachedir) / "annotated_scenegraph.json", "w") as f:
        json.dump(annots, f, indent=4)


def main():
    # Process command-line args (if any)
    args = tyro.cli(ProgramArgs)
    
    # print using masking option
    print(f"args.masking_option: {args.masking_option}")

    if args.mode == "extract-node-captions":
        extract_node_captions(args)
    elif args.mode == "refine-node-captions":
        refine_node_captions(args)
    elif args.mode == "build-scenegraph":
        build_scenegraph(args)
    elif args.mode == "generate-scenegraph-json":
        generate_scenegraph_json(args)
    elif args.mode == "annotate-scenegraph":
        annotate_scenegraph(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
