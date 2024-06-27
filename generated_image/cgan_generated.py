import torch
import numpy as np
import os
import util.utils as utils
from torchvision.utils import save_image

def gen_image(model, opt, device) -> None:
    
    os.makedirs(opt.path_save, exist_ok=True)
    nimage = opt.niter * opt.batch
    
    for j in range(0, nimage, opt.batch):
        sample_y_ = torch.zeros(opt.batch, opt.classes).scatter_(1, torch.randint(0, opt.classes - 1, (opt.batch,  1)).type(torch.LongTensor), 1) #ONE HOT ENCODING 
        sample_z_ = torch.rand((opt.batch, opt.zdim_in))
        sample_z_, sample_y_ = sample_z_.to(device), sample_y_.to(device)
        if opt.type == 'cvae':
            samples = model.decoder(torch.cat((sample_z_, sample_y_), dim=1).to(device))
        elif opt.type == 'vae':
            samples = model.decoder(sample_z_)
        elif 'gan' in opt.type:
            samples = model(sample_z_, sample_y_)
        else:
            raise NotImplementedError(f'{opt.type} is not supported')

        for i, sample in enumerate(samples):
            save_image(sample, f'{opt.path_save}/test_{j+i}.png', nrow=3)
