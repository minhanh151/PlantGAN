
from torchvision import models
import torch.nn as nn

num_classes = 15
model_ft = models.inception_v3(pretrained=True)
# Handle the auxilary net
num_ftrs = model_ft.AuxLogits.fc.in_features
model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
# Handle the primary net
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs,num_classes)
input_size = 299

import torch
state_dict = torch.load("inceptionv3_classifier.pt")
model_ft.load_state_dict(state_dict, strict=True)
model_ft.eval()
# print(model_ft, state_dict)