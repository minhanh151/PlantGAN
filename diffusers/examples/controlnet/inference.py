import torch
from accelerate import PartialState
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
import random
import gc
import os
import argparse
from PIL import Image 
import glob 

BATCH_SIZE = 16
NUM_ITER = 785
# NUM_ITER = 1
FOLDER_CONDITION = "/home/mia/Downloads/DATA/PlantVillage_canny/val"
FOLDER_SAVE = "/home/mia/Downloads/GitHub/PlantGAN/results/gen_controlnet_seg_v1.2_t2i_v5_imgs256"
STABLE_DIFFUSION_PATH="/home/mia/Downloads/GitHub/diffusers/examples/text_to_image/plantVillage_text2image_v5/checkpoint-3000"
# os.makedirs(FOLDER_SAVE, exist_ok=True)


def inference(model, control_net_path, folder_save=FOLDER_SAVE, folder_condition=FOLDER_CONDITION, n_infer_steps=25, atten_scale=0.8, control_scale=0.5):
    # load pipeline
    controlnet = ControlNetModel.from_pretrained(control_net_path, torch_dtype=torch.float16, use_safetensors=True)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    )
    pipeline.load_lora_weights(model, weight_name="pytorch_lora_weights.safetensors", adapter_name="plant")
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    # pipeline.enable_model_cpu_offload()

    # load to distributed state
    distributed_state = PartialState()
    pipeline.to(distributed_state.device)

    # prompt
    prompts = {
        'Pepper': [
            'a bacterial leaf spot with small dark brown water-soaked lesions'
            # 'a healthy leaf with a vibrant green color '
        ],
        'Potato': [
            'a early leaf blight with small dark brown spots that have concentric rings',
            'a late leaf blight featuring irregular dark spots and lighter green halos'
            # 'a healthy leaf with a lush green appearance and no visible signs of disease',
        ],
        'Tomato': [
            'a leaf showing symptoms of bacterial leaf spot with small dark brown lesions that have yellow halos',
            'a leaf showing symptoms of early blight with dark brown spots that have concentric rings and surrounding yellow halos',
            'a late leaf blight with large irregularly shaped dark brown to black lesions pale green or yellowish tissue',
            'a leaf mold with yellow blotches and greyish-brown mould',
            'a leaf septoria spot with numerous small circular dark brown lesions that have grayish centers and surrounded by yellow halos',
            'a leaf two-spot spider mites with visible stippling tiny yellow or white specks and a mottled appearance due to mite feeding',
            'a leaf Target Spot with small circular to oval dark brown to black spots',
            'a Yellow Leaf Curl Virus with a yellowing of the leaf edges',
            'a leaf Mosaic Virus with mottled patterns of light and dark green uneven leaf coloring and a general mosaic-like appearance'
            # 'a healthy leaf with a vibrant green color smooth texture showing strong well-defined veins and overall vitalit',
        ]
    }

    # condition
    condition_image_list = glob.glob(f"{folder_condition}/*/*")

    os.makedirs(folder_save, exist_ok=True) 
    if __name__ == "__main__": 
        with distributed_state.split_between_processes([1,2,3,4]) as prompt:
            index_process = distributed_state.process_index
            for ii in range(NUM_ITER):
                seed = index_process*NUM_ITER + ii
                condition_ii = random.choices(condition_image_list, k=BATCH_SIZE)
                folders = [f.split('/')[-2] for f in condition_ii]
                types_leaf = [folder.split('_')[0] for folder in folders]
                prompt_ii = ["" if 'healthy' in folder else  random.choice(prompts[type_leaf]) for (folder,type_leaf) in zip(folders,types_leaf)]
                # prompt
                # prompt_ii = [""] * len(types_leaf) 
                # print(prompt_ii)
                result = pipeline(prompt_ii, 
                                image=[Image.open(img) for img in condition_ii],
                                num_inference_steps=n_infer_steps,
                                negative_prompt=['anime, do not have background, low quality']*BATCH_SIZE,
                                controlnet_conditioning_scale=control_scale,
                                cross_attention_kwargs={"scale": atten_scale},
                                height=256,
                                width=256,
                                generator=torch.manual_seed(seed)).images
                for j in range(len(result)):
                    # print(result[j])
                    result[j].save(f"{folder_save}/result_{index_process}_{ii}_{j}.png")
                del result
                torch.cuda.empty_cache()
                gc.collect()

def remove_black_image(folder):
    for img in os.listdir(folder):
        path_img = os.path.join(folder, img)
        if os.path.getsize(path_img) <= 842:
            os.remove(path_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--folder_save", type=str, default=FOLDER_SAVE, help='Path to save folder')
    parser.add_argument("--folder_cond", type=str, default=FOLDER_CONDITION, help='Path to condition folder')

    parser.add_argument("--model", type=str, default="plantVillage_text2image_/checkpoint-2500", help="path2lora model")
    parser.add_argument("--control_model", type=str, default="/home/mia/Downloads/GitHub/diffusers/examples/controlnet/plantvillage_seg_v1", help="path2lora model")
    
    parser.add_argument("--n_infer_steps", type=int, default=25, help="number of inference step")
    parser.add_argument('--atten_scale', type=float, default=0.8, help="attention scale")
    parser.add_argument('--cond_scale', type=float, default=0.5, help="guidance scale")
    
    args = parser.parse_args()

    inference(model=args.model, control_net_path=args.control_model, 
              folder_save=args.folder_save, folder_condition=args.folder_cond,
              n_infer_steps=args.n_infer_steps, 
              atten_scale=args.atten_scale, 
              control_scale=args.cond_scale)
    
    remove_black_image(args.folder_save)