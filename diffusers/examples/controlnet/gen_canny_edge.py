from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
import os

IMAGE_FOLDER = "/home/mia/Downloads/DATA/PlantVillage"
IMAGE_CANNY_FOLDER = "/home/mia/Downloads/DATA/PlantVillage_cannny"
os.makedirs(IMAGE_CANNY_FOLDER, exist_ok=True)

def save_canny_image(path, save_path):
    original_image = Image.open(path)

    image = np.array(original_image)

    low_threshold = 200
    high_threshold = 250

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    canny_image.save(save_path)
# original_image.save('ori.png')

if __name__ == '__main__':
    image_extensions = {".jpg", ".jpeg", ".png"}
    for folder, _ , imagepaths in os.walk(IMAGE_FOLDER):
        for image in imagepaths:    
            save_folder = folder.replace("PlantVillage", "PlantVillage_cannny")
            image_path =  os.path.join(folder, image)
            save_path= os.path.join(save_folder, image)
            os.makedirs(save_folder, exist_ok=True)
            
            if os.path.splitext(image)[1].lower() in image_extensions:
                save_canny_image(path=image_path, save_path=save_path)

