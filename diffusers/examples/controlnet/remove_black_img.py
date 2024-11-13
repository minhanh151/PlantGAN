import os 

PATH = '/home/mia/Downloads/GitHub/PlantGAN/results/gen_stable_diffusion_ckpt2500_mse_l1_lpips'
for img in os.listdir(PATH):
    path_img = os.path.join(PATH, img)
    if os.path.getsize(path_img) <= 842:
        os.remove(path_img)