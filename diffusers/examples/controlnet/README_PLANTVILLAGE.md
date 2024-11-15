# ControlNet

## Environment 
- Build môi trường
```bash
conda create -n diffuser python=3.9
conda activate diffuser
conda install -c conda-forge diffusers
pip install diffusers["torch"] transformers
cd diffusers/examples/controlnet
pip install -r requirement.txt

# optimize for memories
pip install bitsandbytes
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121

# prepare validation image 
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png

wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

Chạy môi trường:
`conda activate diffuser`


## Dữ liệu
File meta-data chứa path đến ảnh condition là 
`/home/mia/Downloads/DATA/PlantVillage_canny`

Có hai dataset là canny (canny_v2) và segmentation (canny).  Muốn huấn luyện tập dữ liệu thì đổi tên tập đó thành Plantvillage_canny 

# Command training4PlantVillage 
```
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path/to/save/model"


accelerate launch train_controlnet.py  --pretrained_model_name_or_path=$MODEL_DIR  --output_dir=$OUTPUT_DIR --train_data_dir "/home/mia/Downloads/DATA/PlantVillage"  --resolution=256  --learning_rate=1e-5  --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png"  --validation_prompt "red circle with blue background" "cyan circle with brown floral background"  --train_batch_size=1  --gradient_accumulation_steps=4 --use_8bit_adam   --gradient_checkpointing   --enable_xformers_memory_efficient_attention   --set_grads_to_none --conditioning_image_column "condition" --max_train_steps 15000 --checkpointing_steps 1000
```

**Lưu ý** có thể thay đổi validation image 


## Inference 
### Môi trường 
`conda activate mia`

### Set up accelerate
Chạy `accelerate config` 
**Lưu ý** Khi chạy infer có thể chạy fp16

### Inferece
File chạy inference `/home/mia/Downloads/GitHub/diffusers/examples/controlnet/inference.py`

Câu lệnh chạy 

`accelerate launch --num_processes=2 inference.py --folder_save PATH2SAVE --folder_cond PATH2CONDITIONFOLDER --model PATH2MODEL --controlmodel PATH2ControlNet`


**Lưu ý** 
+ Số lượng processes phải bằng với số GPUs device đã set trong accelerate 