## PlantGAN &mdash; Official Pytorch 

## Implementation
### Normal Installation (Not Recommended)
``` bash
pip install -r requirements.txt
```
Change the model weigth path in the file to the following

`/workspace/PlantGAN/` to  `./` 

For example:
`/workspace/PlantGAN/diffusers/examples/text_to_image/plantVillage_text2image_/checkpoint-2500`


### Docker 
To build image
```
docker build . -t streamlit_env
docker run -td --name streamlit_fid --shm-size 16Gb --network host --gpus all --entrypoint /bin/bash streamlit_env
```

### Streamlit
To run streamlit, demo
```
CUDA_VISIBLE_DEVICES=0,1,2,3 streamlit run app.py
```



## Datasets
We use `PlantVillage` dataset in `datasets/PlantVillage`

## Models
1. Stable Diffusion 
+ train with LORA: `diffusers/examples/text_to_image/plantVillage_text2image_/checkpoint-2500/`
+ train with LORA + perceptual loss: `diffusers/examples/text_to_image/plantVillage_text2image_lpips_l1
/checkpoint-2500/`
+ train with LORA + perceptual loss + loss mse on the latent space: `examples/text_to_image
/plantVillage_text2image_lpips_l1_lr1e-5_l2_2000`
+ train with LORA + perceptual loss + loss mse on the latent space + loss mse on the pixel level:  `/diffusers/examples/text_to_image/plantVillage_text2image_lpips_l1_lr1e-5_l2_2000
/checkpoint-2500/`

2. StyleGAN2: `stylegan2/plantvillage/00005-PlantVillage-cond-auto4
/network-snapshot-023788.pkl`

3. We also retrain inceptionv3 model on the `PlantVillage` dataset to calculate FID: `weights/inceptionv3_classifier.pt`

