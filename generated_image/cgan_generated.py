import torch
import numpy as np
import os
import util.utils as utils

def gen_image(model, opt, device) -> None:
    
    os.makedirs(opt.path_save, exist_ok=True)
    num_iter = opt.nimage // opt.batch + 1

    
    for j in range(0, num_iter, opt.batch):
        sample_y_ = torch.zeros(opt.batch, opt.classes).scatter_(1, torch.randint(0, opt.classes - 1, (opt.batch,  1)).type(torch.LongTensor), 1) #ONE HOT ENCODING 
        sample_z_ = torch.rand((opt.batch, opt.zdim_in))
        sample_z_, sample_y_ = sample_z_.to(device), sample_y_.to(device)

        samples = model(sample_z_, sample_y_)
        samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        samples = (samples + 1) / 2

        for i, sample in enumerate(samples):
            utils.save_images(sample[np.newaxis, ...], [1, 1],
                            f'{opt.path_save}/test_{j+i}.png')


