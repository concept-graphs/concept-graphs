from PIL import Image

def crop_image_pil(image: Image, x1:int, y1:int, x2:int, y2:int, padding:int=0) -> Image:
    '''
    Crop the image with some padding
    
    Args:
        image: PIL image
        x1, y1, x2, y2: bounding box coordinates
        padding: padding around the bounding box
         
    Returns:
        image_crop: PIL image
    '''
    image_width, image_height = image.size
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding)
    
    image_crop = image.crop((x1, y1, x2, y2))
    return image_crop

