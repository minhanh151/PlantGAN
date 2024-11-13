from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
import os
import torch 

# using UperNet4SemanticSegmentation
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

IMAGE_FOLDER = "/home/mia/Downloads/DATA/PlantVillage"
IMAGE_SEG_FOLDER = "/home/mia/Downloads/DATA/PlantVillage_seg"
os.makedirs(IMAGE_SEG_FOLDER, exist_ok=True)

# define models to segment image
image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

# define palette
palette = np.load('palette.npy')

def save_seg_image(path, save_path):
    '''
    generated segmentation of a image and save it to save_path
    '''
    image = Image.open(path).convert('RGB')
    
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)

    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]


    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg.astype(np.uint8)

    image = Image.fromarray(color_seg)
    ################
        # image = np.array(original_image)
    # low_threshold = 200
    # high_threshold = 250

    # image = cv2.Canny(image, low_threshold, high_threshold)
    # image = image[:, :, None]
    # image = np.concatenate([image, image, image], axis=2)
    # seg_image = Image.fromarray(image)
    #
    ##################
    image.save(save_path)
# original_image.save('ori.png')

if __name__ == '__main__':
    image_extensions = {".jpg", ".jpeg", ".png"}
    for folder, _ , imagepaths in os.walk(IMAGE_FOLDER):
        for image in imagepaths:    
            save_folder = folder.replace("PlantVillage", "PlantVillage_seg")
            image_path =  os.path.join(folder, image)
            save_path= os.path.join(save_folder, image)
            os.makedirs(save_folder, exist_ok=True)
            
            if os.path.splitext(image)[1].lower() in image_extensions:
                save_seg_image(path=image_path, save_path=save_path)

