import sys
sys.path.append('stylegan2')
sys.path.append('stylegan2/dnnlib')
import PIL.Image
from diffusers import DiffusionPipeline
import streamlit as st
from diffusers.training_utils import set_seed

import torch
from torchvision import models, transforms
import torch.nn as nn

from datetime import datetime
import PIL 
import numpy as np 
from typing import List, Optional
import stylegan2.dnnlib as dnnlib
import os
import stylegan2.legacy as legacy
from math import ceil
import gc
import time
from collections import Counter
from codecarbon import EmissionsTracker
import pandas as pd 
from evaluation import TrueDataset, fid_score
import cv2
# ===============Define global variable=============================
CLASSIFIER_DEVICE=torch.device('cuda:0')
BATCH_SIZE = 40
BATCH_SIZE_STYLEGAN2 = 20
FOLDER_GT = "/data/PlantVillage/val"
MODEL_LIST = ["StyleGan2",
            "StableDiffusion + LORA", 
            "StableDiffusion + LORA + Perceptual Loss", 
            "StableDiffusion + LORA + Perceptual Loss+ L1", 
            "StableDiffusion + LORA + Perceptual Loss + L1 + L2"]

CLASSES_LIST = [
    'Pepper bell bacterial',
    'Pepper bell healthy',
    'Potato early blight',
    'Potato late blight',
    'Potato healthy',
    'Tomato bacterial spot',
    'Tomato early blight',
    'Tomato late blight',
    'Tomato leaf mold',
    'Tomato septorial leaf spot',
    'Tomato spider mites',
    'Tomato target spot',
    'Tomato yellow curl virus',
    'Tomato mosaic virus',
    'Tomato healthy'
]


CLASS_STYLEGAN_LIST = [
    'Potato early blight',
    'Tomato yellow curl virus',
    'Potato late blight',
    'Pepper bell bacterial',
    'Tomato spider mites', 
    'Tomato bacterial spot', 
    'Tomato target spot',
    'Tomato late blight', 
    'Tomato early blight', 
    'Tomato septorial leaf spot' , 
    'Tomato leaf mold', 
    'Pepper bell healthy',
    'Tomato mosaic virus',
    'Potato healthy', 
    'Tomato healthy', 
]
PROMPT = [
        'bell pepper bacterial leaf spot with small dark brown water-soaked lesions',
        'a healthy bell pepper leaf',
        'potato early leaf blight with small dark brown spots that have concentric rings',
        'potato late leaf blight featuring irregular dark spots and lighter green halos',
        'a healthy potato leaf with a lush green appearance and no visible signs of disease',
        'tomato leaf showing symptoms of bacterial leaf spot with small dark brown lesions that have yellow halos',
        'tomato leaf showing symptoms of early blight with dark brown spots that have concentric rings and surrounding yellow halos',
        'tomato late leaf blight with large irregularly shaped dark brown to black lesions pale green or yellowish tissue',
        'tomato leaf mold with yellow blotches and greyish-brown mould',
        'tomato leaf septoria spot with numerous small circular dark brown lesions that have grayish centers and surrounded by yellow halos',
        'tomato leaf two-spot spider mites with visible stippling tiny yellow or white specks and a mottled appearance due to mite feeding',
        'tomato leaf Target Spot with small circular to oval dark brown to black spots',
        'tomato Yellow Leaf Curl Virus with pronounced leaf curling crinkling and a yellowing of the leaf edges',
        'tomato leaf Mosaic Virus with mottled patterns of light and dark green uneven leaf coloring and a general mosaic-like appearance',
        'healthy tomato leaf with a vibrant green color smooth texture showing strong well-defined veins and overall vitalit',
    ]

CLASS2NAME = [
   'Potato___Early_blight', 
   'Tomato__Tomato_YellowLeaf__Curl_Virus', 
   'Potato___Late_blight', 
   'Pepper__bell___Bacterial_spot', 
   'Tomato_Spider_mites_Two_spotted_spider_mite', 
   'Tomato_Bacterial_spot', 
   'Tomato__Target_Spot', 
   'Tomato_Late_blight', 
   'Tomato_Early_blight', 
   'Tomato_Septoria_leaf_spot', 
   'Tomato_Leaf_Mold', 
   'Pepper__bell___healthy', 
   'Tomato__Tomato_mosaic_virus', 
   'Potato___healthy', 
   'Tomato_healthy',
]
# ===========================================================================

@st.cache_resource()
def load_classfier():
    num_classes = len(CLASSES_LIST)
    model_ft = models.inception_v3(pretrained=True)
    # Handle the auxilary net
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes)

    state_dict = torch.load("weights/inceptionv3_classifier.pt")
    model_ft.load_state_dict(state_dict)
    model_ft.eval()
    return model_ft

@st.cache_resource()
def load_model(path2LoRA):
    pipeline = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
    )
    pipeline.load_lora_weights(path2LoRA, weight_name="pytorch_lora_weights.safetensors", adapter_name="plant")

    return pipeline
    
@st.cache_resource()
def load_gan(path='/workspace/PlantGAN/stylegan2/plantvillage/00005-PlantVillage-cond-auto4/network-snapshot-023788.pkl'):
    #load all the model
    with dnnlib.util.open_url(path) as f:
        stylegan = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    return stylegan


@st.fragment
def display(list_image, list_predict, save_folder=None, save=False):
    controls = st.columns(3)
    with controls[0]:
        batch_size = st.select_slider("Batch size:",range(10,110,10))
    with controls[1]:
        row_size = st.select_slider("Row size:", range(1,6), value = 5)
    num_batches = ceil(len(list_image)/batch_size)
    with controls[2]:
        page = st.selectbox("Page", range(1,num_batches+1))
    batch_img = list_image[(page-1)*batch_size : page*batch_size]
    batch_pred = list_predict[(page-1)*batch_size : page*batch_size]
    grid = st.columns(row_size)
    col = 0
    i = 0
    for image, pred in zip(batch_img, batch_pred):
        with grid[col]:
            if save:
                # img_save = PIL.Image.fromarray(image)
                image.save(f"{save_folder}/{i}.jpg")
            st.image(image, caption=f"Predicted: {CLASSES_LIST[int(pred)]}")
            i += 1
        col = (col + 1) % row_size

# print(pipeline.device)
if __name__ =='__main__':
    
    st.title("Stable Diffusion for leaf generation")
    classifier = load_classfier()
    model_choice = st.selectbox(
                "Stable Diffusion Model version",
                tuple(MODEL_LIST),
                index=None,
                placeholder="Select model...",
            )
    class_choice = st.selectbox(
            "Choose type of leaf",
            tuple(CLASSES_LIST),
            index=None,
            placeholder="Select type of leaf..."
        )
    
    write_file = st.checkbox("Save prediction")
    if write_file:
        cal_fid = st.checkbox("Show FID score")
    else:
        cal_fid = False


    number_of_image = st.slider(
            "number_of_image_gen",
            min_value=5,
            max_value=100,
            value=50,
            step=1,
            label_visibility="visible",
            help="Number of image the models have to gen",
        )

    trans = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) 
    
    
    # define list of image generated
    list_img_gen = []
    list_pred_gen = [] 
    if not model_choice or not class_choice or not number_of_image:
        st.stop()

    


    tracker = EmissionsTracker(allow_multiple_runs=True)
    st.subheader("Parameters", anchor=None)
    
    if "StableDiffusion" in model_choice:
        # ===============StableDiffusionParameters===============================
        num_inference_steps = 25
        with st.expander("Show Params"):
            guidance_scale = st.slider(
                "guidance_scale",
                min_value=0.0,
                max_value=30.0,
                value=7.0,
                step=0.1,
                help="Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality.",
                label_visibility="visible",
            )
            height = st.slider(
                "height",
                min_value=64,
                max_value=1024,
                value=256,
                step=8,
                label_visibility="visible",
            )
            width = st.slider(
                "width",
                min_value=64,
                max_value=1024,
                value=256,
                step=8,
                label_visibility="visible",
            )
            eta = st.slider(
                "eta (η)",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.1,
                help="Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502",
                label_visibility="visible",
            )
            seed = st.slider(
                "seed",
                min_value=0,
                max_value=1024,
                value=10,
                step=1,
                label_visibility="visible",
                help="seed",
            )
            negative_prompt = 'anime, do not have background, fruit, bad quality, low quality, pepper color'
        # =====================================================================


        
        tracker.start()
        set_seed(seed)
        uid = datetime.now().strftime("%Y%m%d_%H:%M:%S")
        submit_button = st.button("Generate", help=None, args=None, kwargs=None)
        my_bar = st.progress(0)


        folder_save = f'results/{model_choice}/{CLASS2NAME[CLASS_STYLEGAN_LIST.index(class_choice)]}_{number_of_image}_{seed}'
        os.makedirs(f'{folder_save}/images', exist_ok=True)

        # load model
        if MODEL_LIST.index(model_choice) == 1:
            pipeline = load_model('/workspace/PlantGAN/diffusers/examples/text_to_image/plantVillage_text2image_/checkpoint-2500')
            device = torch.device('cuda:0')
        elif MODEL_LIST.index(model_choice) == 2:
            pipeline = load_model('/workspace/PlantGAN/diffusers/examples/text_to_image/plantVillage_text2image_perceptual_loss/checkpoint-2500')
            device = torch.device('cuda:1')
        elif MODEL_LIST.index(model_choice) == 3:
            pipeline = load_model('/workspace/PlantGAN/diffusers/examples/text_to_image/plantVillage_text2image_lpips_l1/checkpoint-2500')
            device = torch.device('cuda:2')
        else:
            pipeline = load_model('/workspace/PlantGAN/diffusers/examples/text_to_image/plantVillage_text2image_lpips_l1_lr1e-5_l2_2000/checkpoint-2500')
            device = torch.device('cuda:3')
        
        text_prompt = PROMPT[CLASSES_LIST.index(class_choice)]
        classifier.to(CLASSIFIER_DEVICE)
        pipeline.to(device)
 
        model_loading_emssion: float =  tracker.stop()
        
        if not submit_button:
            # print(submit_button)
            st.stop()

        tracker.start()
        
        st.subheader("Output", anchor=None)
        start_time = time.time()
        
        
        for ii in range(0, number_of_image, BATCH_SIZE):
            # TODO set different seed 
            num_images_per_prompt = min(BATCH_SIZE, number_of_image - ii)
            result = pipeline(text_prompt, num_inference_steps=num_inference_steps, cross_attention_kwargs={"scale": 0.8}, 
                                guidance_scale=guidance_scale, height=height, width=width, num_images_per_prompt=num_images_per_prompt, 
                                negative_prompt=negative_prompt).images
        
            list_img_gen.extend(result)
            
            # predict class
            res_tensor = torch.cat([trans(img).unsqueeze(0) for img in result], dim=0)
            output = classifier(res_tensor.to(CLASSIFIER_DEVICE))
            _, list_cls_idx = torch.max(output, dim=1)
            
            list_pred_gen.extend(list_cls_idx)

            del res_tensor
            del output
            torch.cuda.empty_cache()
            gc.collect()

            list_pred_gen = [int(pred) for pred in list_pred_gen]
            list_gt = [CLASSES_LIST.index(class_choice)] * number_of_image
            positive_sample = len(list((Counter(list_pred_gen) & Counter(list_gt)).elements()))
        

        infer_emssion: float = tracker.stop()
        st.text(f"Running time: {time.time()- start_time:.2f}s")
        st.text(f"True Predict: {positive_sample}, Negative Predict: {number_of_image - positive_sample}, Accuracy: {int(positive_sample/number_of_image * 100)}%")
        display(list_image=list_img_gen, list_predict=list_pred_gen, save_folder= f'{folder_save}/images', save=write_file)
        if cal_fid:
            # cal fid 
            folder_gt = f'{FOLDER_GT}/{CLASS2NAME[CLASS_STYLEGAN_LIST.index(class_choice)]}/'
            # print(folder_gt)
            true_dataset = TrueDataset(folder_gt)
            # print(len(true_dataset))
            fake_dataset = TrueDataset(f'{folder_save}/images')
            # print(len(fake_dataset))
            fid = fid_score(true_dataset, fake_dataset, device)
            st.text(f"FID_SCORE: {fid:.2f}")
        else:
            fid = None

        if write_file:
            df = pd.DataFrame({
                "True Predict": positive_sample,
                "Negative Predict": number_of_image - positive_sample,
                "Accuracy": positive_sample/number_of_image,
                "FID": fid,
                'Emission': infer_emssion,
            },  index=[0]).to_csv(f"{folder_save}/result.csv")

        
        st.text(f"Emission from loading: {model_loading_emssion* 1e8:.2f} * 10^-8 kg.EQ.CO2") 
        st.text(f"Emission from inference: {infer_emssion * 1e4:.4f} * 10^-4 kg.EQ.CO2")
        
       
        
        
            
    elif "StyleGan" in model_choice:
        with st.expander("Show params"):
            seed = st.slider(
                "seed",
                min_value=0,
                max_value=100,
                value=10,
                step=1,
                label_visibility="visible",
                help="seed",
            )
            truncation_psi = st.slider(
                "truncation_psi",
                min_value=0,
                max_value=5,
                value=1,
                step=1,
                label_visibility="visible",
                help="seed",
            )
            noise_mode = st.selectbox(
                "Choose type of noise",
                ('const', 'random', 'none'),
            )
            device = torch.device('cuda:0')
            
            tracker.start()
            G = load_gan()
            classifier.to(device)
            model_loading_emssion: float =  tracker.stop()
        
        submit_button = st.button("Generate", help=None, args=None, kwargs=None)
        my_bar = st.progress(0)
        if not submit_button:
            st.stop()
        
        

        folder_save = f'results/{model_choice}/{CLASS2NAME[CLASS_STYLEGAN_LIST.index(class_choice)]}_{number_of_image}_{seed}'
        os.makedirs(f'{folder_save}/images', exist_ok=True)

    
        st.subheader("Output", anchor=None)
        start_time = time.time()

        tracker.start()
        # ===gen_image========
        for ii in range(0, number_of_image, BATCH_SIZE_STYLEGAN2):
            label = torch.zeros([BATCH_SIZE_STYLEGAN2, G.c_dim], device=device)
            label[:, CLASS_STYLEGAN_LIST.index(class_choice)] = 1
            z = torch.from_numpy(np.random.RandomState( + ii).randn(BATCH_SIZE_STYLEGAN2, G.z_dim)).to(device)
            img = G(z, label, truncation_psi=1, noise_mode=noise_mode)
            img_show = (img.clone().permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_show = np.array(img_show.cpu())
            img_show = [PIL.Image.fromarray(im) for im in img_show]
        #====================
            list_img_gen.extend(img_show)
            img = torch.nn.functional.interpolate(img, size=299)
            output = classifier(img)
            conf, cls_idx = torch.max(output, dim=1)
            list_pred_gen.extend(cls_idx)
            torch.cuda.empty_cache()
        
        list_pred_gen = [int(pred) for pred in list_pred_gen]
        list_gt = [CLASSES_LIST.index(class_choice)] * number_of_image
        positive_sample = len(list((Counter(list_pred_gen) & Counter(list_gt)).elements()))
        infer_emssion: float = tracker.stop()
        
        st.text(f"Running time: {time.time()- start_time:.2f}s")
        st.text(f"True Predict: {positive_sample}, Negative Predict: {number_of_image - positive_sample}, Accuracy: {int(positive_sample/number_of_image * 100)}%")
        st.text(f"Emission from loading: {model_loading_emssion* 1e8:.2f} * 10^-8 Wh, Emission from inference: {infer_emssion * 1e4:.4f} * 10^-4 Wh")
        display(list_image=list_img_gen, list_predict=list_pred_gen,  save_folder= f'{folder_save}/images', save=write_file)


        if cal_fid:
            # cal fid 
            folder_gt = f'{FOLDER_GT}/{CLASS2NAME[CLASS_STYLEGAN_LIST.index(class_choice)]}/'
            # print(folder_gt)
            true_dataset = TrueDataset(folder_gt)
            # print(len(true_dataset))
            fake_dataset = TrueDataset(f'{folder_save}/images')
            # print(len(fake_dataset))
            fid = fid_score(true_dataset, fake_dataset, device)
            st.text(f"FID_SCORE: {fid:.2f}")
        else:
            fid = None

        if write_file:
            df = pd.DataFrame({
                "True Predict": positive_sample,
                "Negative Predict": number_of_image - positive_sample,
                "Accuracy": positive_sample/number_of_image,
                "FID": fid,
                'Emission': infer_emssion,
            },  index=[0]).to_csv(f"{folder_save}/result.csv")
        
        