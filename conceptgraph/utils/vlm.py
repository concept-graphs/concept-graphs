import json
from openai import OpenAI
import os
import base64

from PIL import Image
import numpy as np

import ast
import re

system_prompt_1 = '''
You are an agent specialized in describing the spatial relationships between objects in an annotated image.

You will be provided with an annotated image and a list of labels for the annotations. Your task is to determine the spatial relationships between the annotated objects in the image, and return a list of these relationships in the correct list of tuples format as follows:
[("object1", "spatial relationship", "object2"), ("object3", "spatial relationship", "object4"), ...]

Your options for the spatial relationship are "on top of" and "next to".

For example, you may get an annotated image and a list such as 
["cup 3", "book 4", "clock 5", "table 2", "candle 7", "music stand 6", "lamp 8"]

Your response should be a description of the spatial relationships between the objects in the image. 
An example to illustrate the response format:
[("book 4", "on top of", "table 2"), ("cup 3", "next to", "book 4"), ("lamp 8", "on top of", "music stand 6")]
'''

'''
You are an agent specialized in identifying and describing objects that are placed "on top of" each other in an annotated image. You always output a list of tuples that describe the "on top of" spatial relationships between the objects, and nothing else. When in doubt, output an empty list.

When provided with an annotated image and a corresponding list of labels for the annotations, your primary task is to determine and return the "on top of" spatial relationships between the annotated objects. Your responses should be formatted as a list of tuples, specifically highlighting objects that rest on top of others, as follows:
[("object1", "on top of", "object2"), ...]
'''

# Only deal with the "on top of" relation
system_prompt_only_top = '''
You are an agent specializing in identifying the physical and spatial relationships in annotated images for 3D mapping.

In the images, each object is annotated with a bright numeric id (i.e. a number) and a corresponding colored contour outline. Your task is to analyze the images and output a list of tuples describing the physical relationships between objects. Format your response as follows: [("1", "relation type", "2"), ...]. When uncertain, return an empty list.

Note that you are describing the **physical relationships** between the **objects inside** the image.

You will also be given a text list of the numeric ids of the objects in the image. The list will be in the format: ["1: name1", "2: name2", "3: name3" ...], only output the physical relationships between the objects in the list.

The relation types you must report are:
- phyically placed on top of: ("object x", "on top of", "object y") 
- phyically placed underneath: ("object x", "under", "object y") 

An illustrative example of the expected response format might look like this:
[("object 1", "on top of", "object 2"), ("object 3", "under", "object 2"), ("object 4", "on top of", "object 3")]. Do not put the names of the objects in your response, only the numeric ids.

Do not include any other information in your response. Only output a parsable list of tuples describing the given physical relationships between objects in the image.
'''

# For captions
system_prompt_captions = '''
You are an agent specializing in accurate captioning objects in an image.

In the images, each object is annotated with a bright numeric id (i.e. a number) and a corresponding colored contour outline. Your task is to analyze the images and output in a structured format, the captions for the objects.

You will also be given a text list of the numeric ids and names of the objects in the image. The list will be in the format: ["1: name1", "2: name2", "3: name3" ...]

The names were obtained from a simple object detection system and may be inaacurate.

Your response should be in the format of a list of dictionaries, where each dictionary contains the id, name, and caption of an object. Your response will be evaluated as a python list of dictionaries, so make sure to format it correctly. An example of the expected response format is as follows:
[
    {"id": "1", "name": "object1", "caption": "concise description of the object1 in the image"},
    {"id": "2", "name": "object2", "caption": "concise description of the object2 in the image"},
    {"id": "3", "name": "object3", "caption": "concise description of the object3 in the image"}
    ...
]

And each caption must be a concise description of the object in the image.
'''

system_prompt_consolidate_captions = '''
You are an agent specializing in consolidating multiple captions for the same object into a single, clear, and accurate caption.

You will be provided with several captions describing the same object. Your task is to analyze these captions, identify the common elements, remove any noise or outliers, and consolidate them into a single, coherent caption that accurately describes the object.

Ensure the consolidated caption is clear, concise, and captures the essential details from the provided captions.

Here is an example of the input format:
[
    {"id": "3", "name": "cigar box", "caption": "rectangular cigar box on the side cabinet"},
    {"id": "9", "name": "cigar box", "caption": "A small cigar box placed on the side cabinet."},
    {"id": "7", "name": "cigar box", "caption": "A small cigar box is on the side cabinet."},
    {"id": "8", "name": "cigar box", "caption": "Box on top of the dresser"},
    {"id": "5", "name": "cigar box", "caption": "A cigar box placed on the dresser next to the coffeepot."},
]

Your response should be a JSON object with the format:
{
    "consolidated_caption": "A small rectangular cigar box on the side cabinet."
}

Do not include any additional information in your response.
'''

system_prompt = system_prompt_only_top

# gpt_model = "gpt-4-vision-preview"
gpt_model = "gpt-4o-2024-05-13"

def get_openai_client():
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY')
    )
    return client

# Function to encode the image as base64
def encode_image_for_openai(image_path: str, resize = False, target_size: int=512):
    print(f"Checking if image exists at path: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not resize:
        # Open the image
        print(f"Opening image from path: {image_path}")
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            print("Image encoded in base64 format.")
        return encoded_image
    
    print(f"Opening image from path: {image_path}")
    with Image.open(image_path) as img:
        # Determine scaling factor to maintain aspect ratio
        original_width, original_height = img.size
        print(f"Original image dimensions: {original_width} x {original_height}")
        
        if original_width > original_height:
            scale = target_size / original_width
            new_width = target_size
            new_height = int(original_height * scale)
        else:
            scale = target_size / original_height
            new_height = target_size
            new_width = int(original_width * scale)

        print(f"Resized image dimensions: {new_width} x {new_height}")

        # Resizing the image
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        print("Image resized successfully.")
        
        # Convert the image to bytes and encode it in base64
        with open("temp_resized_image.jpg", "wb") as temp_file:
            img_resized.save(temp_file, format="JPEG")
            print("Resized image saved temporarily for encoding.")
        
        # Open the temporarily saved image for base64 encoding
        with open("temp_resized_image.jpg", "rb") as temp_file:
            encoded_image = base64.b64encode(temp_file.read()).decode('utf-8')
            print("Image encoded in base64 format.")
        
        # Clean up the temporary file
        os.remove("temp_resized_image.jpg")
        print("Temporary file removed.")

    return encoded_image

def consolidate_captions(client: OpenAI, captions: list):
    # Formatting the captions into a single string prompt
    captions_text = "\n".join([f"{cap['caption']}" for cap in captions if cap['caption'] is not None])
    user_query = f"Here are several captions for the same object:\n{captions_text}\n\nPlease consolidate these into a single, clear caption that accurately describes the object."

    messages = [
        {
            "role": "system",
            "content": system_prompt_consolidate_captions
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    consolidated_caption = ""
    try:
        response = client.chat.completions.create(
            model=f"{gpt_model}",
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        consolidated_caption_json = response.choices[0].message.content.strip()
        consolidated_caption = json.loads(consolidated_caption_json).get("consolidated_caption", "")
        print(f"Consolidated Caption: {consolidated_caption}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        consolidated_caption = ""

    return consolidated_caption
    
def extract_list_of_tuples(text: str):
    # Pattern to match a list of tuples, considering a list that starts with '[' and ends with ']'
    # and contains any characters in between, including nested lists/tuples.
    text = text.replace('\n', ' ')
    pattern = r'\[.*?\]'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    if match:
        # Extract the matched string
        list_str = match.group(0)
        try:
            # Convert the string to a list of tuples
            result = ast.literal_eval(list_str)
            if isinstance(result, list):  # Ensure it is a list
                return result
        except (ValueError, SyntaxError):
            # Handle cases where the string cannot be converted
            print("Found string cannot be converted to a list of tuples.")
            return []
    else:
        # No matching pattern found
        print("No list of tuples found in the text.")
        return []
    
def vlm_extract_object_captions(text: str):
    # Replace newlines with spaces for uniformity
    text = text.replace('\n', ' ')
    
    # Pattern to match the list of objects
    pattern = r'\[(.*?)\]'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    if match:
        # Extract the matched string
        list_str = match.group(0)
        try:
            # Try to convert the entire string to a list of dictionaries
            result = ast.literal_eval(list_str)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            # If the whole string conversion fails, process each element individually
            elements = re.findall(r'{.*?}', list_str)
            result = []
            for element in elements:
                try:
                    obj = ast.literal_eval(element)
                    if isinstance(obj, dict):
                        result.append(obj)
                except (ValueError, SyntaxError):
                    print(f"Error processing element: {element}")
            return result
    else:
        # No matching pattern found
        print("No list of objects found in the text.")
        return []
    
def get_obj_rel_from_image_gpt4v(client: OpenAI, image_path: str, label_list: list):
    # Getting the base64 string
    base64_image = encode_image_for_openai(image_path)
    
    global system_prompt
    global gpt_model
    
    user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please describe the spatial relationships between the objects in the image."
    
    
    vlm_answer = []
    try:
        response = client.chat.completions.create(
            model=f"{gpt_model}",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt_only_top
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ]
        )
        
        vlm_answer_str = response.choices[0].message.content
        print(f"Line 113, vlm_answer_str: {vlm_answer_str}")
        
        vlm_answer = extract_list_of_tuples(vlm_answer_str)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        vlm_answer = []
    print(f"Line 68, user_query: {user_query}")
    print(f"Line 97, vlm_answer: {vlm_answer}")
    
    
    return vlm_answer

    
def get_obj_captions_from_image_gpt4v(client: OpenAI, image_path: str, label_list: list):
    # Getting the base64 string
    base64_image = encode_image_for_openai(image_path)
    
    global system_prompt
    

    user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please accurately caption the objects in the image."
    
    messages=[
        {
            "role": "system",
            "content": system_prompt_captions
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    
    vlm_answer_captions = []
    try:
        response = client.chat.completions.create(
            model=f"{gpt_model}",
            messages=messages
        )
        
        vlm_answer_str = response.choices[0].message.content
        print(f"Line 113, vlm_answer_str: {vlm_answer_str}")
        
        vlm_answer_captions = vlm_extract_object_captions(vlm_answer_str)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        vlm_answer_captions = []
    print(f"Line 68, user_query: {user_query}")
    print(f"Line 97, vlm_answer: {vlm_answer_captions}")
    
    
    return vlm_answer_captions