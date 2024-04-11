import openai
from openai import OpenAI
import os
import base64
import requests

# Function to encode the image as base64
def encode_image_for_openai(image_path: str):
    # check if the image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Getting the base64 string
img_path = '/home/kuwajerw/new_local_data/new_record3d/ali_apartment/apt_scan_no_smooth_processed/exps/s_detections_stride1_69/vis/160_for_vlm.jpg'
base64_image = encode_image_for_openai(str(img_path))


# Initialize OpenAI client with the new format
client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY')
)


system_prompt = '''
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

def analyze_image():
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
            "content": "Return the spatial relationships between the objects in the image."
        }
    ],
        max_tokens=300,
        top_p=0.1
    )

    return response.choices[0].message.content

result = analyze_image()
print(result)