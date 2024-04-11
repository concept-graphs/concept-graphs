
# WORKS, LAVA 1.5 13B, 4 BIT PRESCSION
import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-13b-hf"


# image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    load_in_8bit=True
)

processor = AutoProcessor.from_pretrained(model_id)

# raw_image = Image.open(requests.get(image_file, stream=True).raw)

img_path = '/home/kuwajerw/new_local_data/new_record3d/ali_apartment/apt_scan_no_smooth_processed/exps/s_detections_stride1_69/vis/160_for_vlm.jpg'
raw_image = Image.open(img_path).convert('RGB')

labels = ['power outlet 1', 'backpack 2', 'computer tower 3', 'poster 4', 'desk 5', 'picture 6', 'bowl 7', 'folded chair 8', 'trash bin 9', 'tissue box 10']

example_labels = []

# inputs = processor(prompt, raw_image, return_tensors='pt').to("cuda", torch.float16)
system_prompt = "What follows is a chat between a human and an artificial intelligence assistant. The assistant always answers the question in the required format"
example_prompt = ""
user_prompt = f"In this picture, there are these annotated objects, labels: {labels}. Are any of these annotated objects (not the labels, the objects) on top of one another in this picture?"
prompt = f"{system_prompt} USER: <image>\n{user_prompt} ASSISTANT:"
print(f"Line 263, prompt: {prompt}")
inputs = processor(prompt, raw_image, return_tensors='pt')

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
k=1
 