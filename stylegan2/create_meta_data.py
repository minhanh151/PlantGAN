import json 
import os

label_str2int = {}
index = 0
labels = []
FOLDER = '/home/mia/Downloads/GitHub/PlantGAN/results/gen_stable_diffusion'
for dirpath, dirnames, filenames in os.walk(FOLDER):
    for filename in filenames:
        if filename.split('.')[-1] in ('JPG', 'jpg', 'jpeg'):
            list_fold_path = dirpath.split('/')
            dir_name = '/'.join(list_fold_path[-2:])
            name = os.path.join(dir_name, filename)
            
            label = list_fold_path[-1]
            if label not in label_str2int.keys():
                label_str2int[label] = index
                index += 1
            
            label_item = [name, label_str2int[label]]
            labels.append(label_item)
        else: print(filename)
print(len(labels))
with open (f'{FOLDER}/dataset.json', 'w') as l:
    json.dump({"labels": labels}, l, indent=2)
            # print(os.path.join(dirpath, filename))