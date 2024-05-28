import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import json
import math
import numpy as np
import textwrap


def wrap_text(text, max_length=30):
    return "\n".join(textwrap.wrap(text, max_length))


def main(source: Path):
    target = source / "viz"
    os.makedirs(target, exist_ok=True)

    # Load captions
    caption_path = source / "cfslam_gpt-4_responses"
    caption_file_names = os.listdir(caption_path)
    object_ids = [int(f.split(".")[0]) for f in caption_file_names]

    # Create a visualization figure for each object
    for object_id in object_ids:
        # Load caption
        caption_file = caption_path / f"{object_id}.json"
        with caption_file.open() as f:
            caption_dict = json.load(f)
        llava_captions = caption_dict["captions"]
        gpt_response = json.loads(caption_dict["response"])
        is_invalid = "object_tag" not in gpt_response or "summary" not in gpt_response or gpt_response["object_tag"] == "invalid"
        if is_invalid:
            gpt_summary = "Invalid object"
        else:
            gpt_summary = gpt_response["summary"]

        if is_invalid:
            print("Invalid captions")
        else:
            print(gpt_summary)

        # Load images
        object_image_path = source / "images" / "processed" / str(object_id)
        object_images_file_names = os.listdir(object_image_path)

        no_grid = not len(object_images_file_names) or not len(llava_captions)

        if no_grid:
            # Empty image
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.axis('off')
        else:
            n = min(9, len(llava_captions))  # Only plot up to 9 images
            nrows = int(math.ceil(n / 3))
            ncols = 3 if n > 1 else 1
            fig, axarr = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), squeeze=False)  # Adjusted figsize
            axarr = axarr.flatten()

            for caption, image_path, ax in zip(llava_captions, object_images_file_names, axarr):
                image = Image.open(object_image_path / image_path)
                ax.imshow(np.array(image))
                # Wrap caption to 30 characters
                caption = wrap_text(caption, 25)
                ax.set_title("\n" + caption, fontsize=16)

            for a in axarr:
                a.axis('off')

        gpt_summary = wrap_text(gpt_summary, 30)
        plt.suptitle("GPT: " + gpt_summary, fontsize=32)
        plt.tight_layout()
        plt.savefig(target / f"{object_id}.png")
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize captions')
    parser.add_argument('source', type=Path, help='Path to the data directory. sg_cache in the CG code.')
    args = parser.parse_args()
    main(Path(args.source))
