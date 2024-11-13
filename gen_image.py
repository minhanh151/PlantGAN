import util.utils as utils
import numpy as np
import os 
import argparse
import torch 

from generated_image.cgan_generated import gen_image
from models.cgan_model import generator 
from cvae_train import Model
from vae_train import VAE

def GenOptions():
    parser = argparse.ArgumentParser(description='some params for generating image')
    parser.add_argument('--weight', type=str, default='checkpoints/plantvillage_cgan/cganlatest_G.pkl',
                         help='path to the generator weight')
    parser.add_argument('--zdim-in', type=int, default=62, 
                        help='number of z channel for input dim, 32 for cvae and vae')
    parser.add_argument('--input-size', type=int, default=64, help='size of input')
    parser.add_argument('--classes', type=int, default=15, help='number of classes of the dataset')
    parser.add_argument('--zdim-out',type=int, default=3, help='number of classes of the output dim')
    parser.add_argument('--batch', type=int, default=225, help='batch size during generation')
    parser.add_argument('--path-save', type=str, default='results', help='path save inference image')
    parser.add_argument('--niter', type=int, default=45, help='number of iteration generated')
    parser.add_argument('--type', type=str, default='vae', help='type of model to infer')

    return parser 

if __name__ == '__main__':
    opt = GenOptions().parse_args()
    print(opt)
    # check device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load model 
    assert os.path.isfile(opt.weight), f'{opt.weight} is not a valid file'
    state_dict = torch.load(opt.weight)
    
    if opt.type == 'cvae':
        model = Model()
    elif opt.type == 'vae':
        model = VAE((3,64,64))
    elif 'gan' in opt.type:
        model = generator(input_dim=opt.zdim_in, output_dim=opt.zdim_out, 
                      input_size=opt.input_size, class_num=opt.classes)
    else:
        raise NotImplementedError(f'{opt.type} is not supported')
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    

    gen_image(model=model, opt=opt, device=device)
