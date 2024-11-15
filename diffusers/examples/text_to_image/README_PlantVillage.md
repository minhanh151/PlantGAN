LORA

# Môi trường
`conda activate mia`

## Dữ liệu

Vị trí dữ liệu `/home/mia/Downloads/DATA/PlantVillage`

Cấu trúc dữ liệu 
- folder
	- train
		- folder_type_leaf_1
			- *.jpg
		- folder_type_leaf_2
		- ....
		- metadata.csv
	- val
		- folder_type_leaf_1
		- folder_type_leaf_2
		- ....
		- metadata.csv


## Training 
### Set up môi trường accelerate config
1. Chạy câu lệnh `accelerate config`. 
```bash
In which compute environment are you running?
This machine                                                                                                         
Which type of machine are you using?                                                                                 
multi-GPU                                                                                                            
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1                           
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: no                                                                                                         
Do you wish to optimize your script with torch dynamo?[yes/NO]:no                                                    
Do you want to use DeepSpeed? [yes/NO]: no                                                                           
Do you want to use FullyShardedDataParallel? [yes/NO]: no                                                            
Do you want to use Megatron-LM ? [yes/NO]: no                                                                        
How many GPU(s) should be used for distributed training? [1]:2
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:2,3
```

**Lưu ý**: 
+ khi training LORA: chỉ chạy được fp32 không chạy được fp16 
+ không sử dụng thêm bất kì phương pháp tối ưu nào khi chạy training diffusion 
### Huấn luyện mô hình
1. Chạy lệnh `train.sh` (`/home/mia/Downloads/GitHub/diffusers/examples/text_to_image/train.sh)

**Lưu ý** 
+ Muốn chạy nhiều training command cùng lúc phải thay đổi `main_process_port` trong `train.sh`
+ Sử dụng perceptual loss thêm args `--add_perceptual_loss`
+ Sử dụng loss_1 thêm args `--add_experiment_loss_1`
+ Sử dụng loss_2 thêm args `--add_experiment_loss_2`


## Inference 
### Set up accelerate
Chạy `accelerate config` 


**Lưu ý** Khi chạy infer có thể chạy fp16

### Inferece
File chạy inference `/home/mia/Downloads/GitHub/diffusers/examples/text_to_image/inference.py`
Câu lệnh chạy `accelerate launch --num_processes=2 inference.py --folder_save PATH2FOLDER --model PATH2MODEL`


**Lưu ý** 
+ Số lượng processes phải bằng với số GPUs device đã set trong accelerate 