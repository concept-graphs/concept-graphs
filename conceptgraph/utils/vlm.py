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

You will also be given a text list of the numeric ids of the objects in the image. The list will be in the format: ["object 1", "object 2", "object 3" ...], only output the physical relationships between the objects in the list.

The relation types you must report are:
- phyically placed on top of: ("object x", "on top of", "object y") 
- phyically placed underneath: ("object x", "under", "object y") 

An illustrative example of the expected response format might look like this:
[("object 1", "on top of", "object 2"), ("object 3", "under", "object 2"), ("object 4", "on top of", "object 3")]

Do not include any other information in your response. Only output a parsable list of tuples describing the given physical relationships between objects in the image.
'''

system_prompt = system_prompt_only_top

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
    
def get_obj_rel_from_image_gpt4v(client: OpenAI, image_path: str, label_list: list):
    # Getting the base64 string
    base64_image = encode_image_for_openai(image_path)
    
    global system_prompt
    
    user_query = f"Here is the list of labels for the annotations of the objects in the image: {label_list}. Please describe the spatial relationships between the objects in the image."
    
    
    vlm_answer = []
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
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