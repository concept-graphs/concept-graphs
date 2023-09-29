class GPTPrompt:
    def __init__(self):
        
        self.old_system_prompt = """
        The input is a list of JSONs describing multiple predictions of a single object. Each JSON has four fields: 
        1. id: a unique identifier for the object
        2. bbox_extent: the 3D bounding box extents of the object 
        3. bbox_center: the 3D bounding box center of the object 
        4. caption: a caption predicted by an image captioning model referencing that object. This caption should be brief and in sparse prose. For example, the caption "the object described appears to be described as a electric bicycle, it is sitting alongside a red suitcase which is nearby" should be shortened to "electric bike, near red suitcase".
        There may be upto 10 such bounding boxes and captions in each input.
        The captions may not always be accurate or consistent (often, predictions may just be wrong). 
        The only valid objects are ones that we commonly find in indoor scenes. Any predictions that reference 
        people, animals, or objects that are impossible to find indoors, must be tagged as "invalid".

        Output a brief, informative language tag for each object being referenced to. If the captions are grossly
        inconsistent, output "invalid" for that object.

        The output must be a single JSON containing just the following fields. "summary" indicating a brief 
        summary of your understanding of the object being referenced to. "possible_tags" indicating a list of 
        possible tags that you think the object could be. "object_tag" indicating the final tag that you think 
        this object should be, considering everything else in the scene (particularly, nearby other objects). 
        Before suggesting the final tag, consider the actual size of the object (you have this in the "bbox") 
        and identify the best possible tag this object could be. Verify that the output is in valid JSON format with
        the fields "summary", "possible_tags", and "object_tag". The object_tag must be supported by the captions.
        It is very important that the output must be valid JSON, and nothing else.
        """
        
        self.new_but_long_system_prompt = """You are a helpful assistant that helps identify and describe objects in a scene. Your input is a in JSON format, and you should always reply in JSON format. Your input will contain the field "caption" which is a list of captions of an image attempting to identify the objects in the image. Your response should contain the fields "summary" containing a concise summary of the object(s) in the image. If an object is mentioned more than once, that prediction is likely accurate. If many objects are mentioned more than once, and a container or surface is mentioned, it is likely that the image was of those objects (mentioned more than once) on that container or surface. If no object is mentioned more than once, and each caption is unrelated to the rest, it could be a blank or too small image or blurry image, the captions are likely incorrect, so just say "conflicting captions about [objects] in the summary field and put "invalid" in the "object_tag" field. The field "possible_tags" should contain a list of possible tags that you think the object(s) could be. The field "object_tag" should contain the final tag that you think this object should be, considering all the information given. These are based on scans of indoor scenes so most objects will be those found in indoor spaces. """
        
        self.system_prompt = """Identify and describe objects in scenes. Input and output must be in JSON format. The input field 'captions' contains a list of image captions aiming to identify objects. Output 'summary' as a concise description of the identified object(s). An object mentioned multiple times is likely accurate. If various objects are repeated and a container/surface is noted such as a shelf or table, assume the (repeated) objects are on that container/surface. For unrelated, non-repeating (or empty) captions, summarize as 'conflicting (or empty) captions about [objects]' and set 'object_tag' to 'invalid'. Output 'possible_tags' listing potential object categories. Set 'object_tag' as the conclusive identification. Focus on indoor object types, as the input captions are from indoor scans."""
        
        self.example_1 = """
        "id": 1,
        "captions": [
        "a jacket hanging on a wall, either on a hook or a rack.",
        "a jacket, which is hanging on a wall or a rack.",
        "a jacket, which is either being worn or draped over a person's shoulders.",
        "a sweater, which is hanging on a clothes hanger.",
        "a hooded jacket, which is either hanging on a hook or draped over a shower rail.",
        "a mannequin, which is wearing a yellow shirt and a red jacket.",
        "a jacket, which is hanging on a hook or a rack.",
        "a hooded sweatshirt, which is either being held by someone or hanging on a hook or a rack.",
        "a hanger with a yellow and black jacket hanging on it.",
        "a yellow and black striped umbrella.",
        "a hanging coat, which is either yellow or red depending on the specific description.",
        "a cat.",
        "a person wearing a yellow and red jacket."
        ]
        }"
        """
        
        self.response_1 = """{
        "summary": "a jacket hanging on a rack",
        "possible_tags": ["jacket", "sweater", "hooded jacket", "hooded sweatshirt", "coat", "hanging clothing", "hanger", "wall", "rack"],
        "object_tag": "hanging jacket"
        }"""
        
        self.example_2 = """{
        "id": 12,
        "captions": [
        "a bookshelf filled with books",
        "a bicycle helmet"
        ]
        }"""
        
        self.response_2 = """{
        "summary": "conflicting captions of a bookshelf and a bicycle helmet",
        "possible_tags": ["bookshelf", "bicycle helmet", "helmet", "bookcase", "book", "books", "shelf", "shelves"],
        "object_tag": "invalid"
        }"""
        
        self.example_3 = """{
        "id": 304,
        "captions": [
        "a pair of scissors.",
        "a sewing machine.",
        "a white shelf or rack, which is filled with various boxes and files.",
        "a white shelf or bookshelf that is filled with various items."
        ]
        }"""
        
        self.response_3 = """{
        "summary": "a white shelf or rack filled with various items",
        "possible_tags": ["shelf", "rack", "bookshelf", "bookcase", "box", "file", "scissors", "sewing machine"],
        "object_tag": "shelf"
        }"""
        
        self.example_4 = """{
        "id": 433,
        "captions": [
        "a white toilet.",
        "a pile of various exercise equipment, including a set of tennis balls.",
        "a white table.",
        "a barbell.",
        "a microwave.",
        "a tall, thin black bottle.",
        "a barbell, which is situated on a table.",
        "a pile of various exercise equipment, including a bench, situated in a room with desks and a whiteboard.",
        "a tall vase or pitcher.",
        "a tennis ball.",
        "a shirt with a space design on it.",
        "a tennis ball.",
        "a gray shirt with the NASA logo on it."
        ]
        }"""
        
        self.response_4 = """{
        "summary": "a white table containing a barbell and possibly with some other exercise equipment",
        "possible_tags": ["exercise equipment", "tennis ball", "barbell", "bench", "table", "microwave", "bottle", "vase", "pitcher", "shirt"],
        "object_tag": "white table"
        }"""
        
        self.example_5 = """{
        "id": 231,
        "captions": [
            "a teddy bear.",
            "a doorknob.",
            "a television set.",
            "a laptop computer."
        ]
        }"""
        
        self.response_5 = """{
        "summary": "conflicting captions of a teddy bear, a doorknob, a television set, and a laptop computer",
        "possible_tags": ["teddy bear", "doorknob", "television set", "laptop computer", "computer"],
        "object_tag": "invalid"
        }"""


    def get_json(self):
        prompt_json = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": self.example_1
            },
            {
                "role": "assistant",
                "content": self.response_1
            },
            {
                "role": "user",
                "content": self.example_2
            },
            {
                "role": "assistant",
                "content": self.response_2
            },
            {
                "role": "user",
                "content": self.example_3
            },
            {
                "role": "assistant",
                "content": self.response_3
            },
            {
                "role": "user",
                "content": self.example_4
            },
            {
                "role": "assistant",
                "content": self.response_4
            },
            {
                "role": "user",
                "content": self.example_5
            },
            {
                "role": "assistant",
                "content": self.response_5
            }
        ]
        return prompt_json

# Usage example
if __name__ == "__main__":
    prompt_obj = GPTPrompt()
    json_data = prompt_obj.get_json()
    print(json_data)