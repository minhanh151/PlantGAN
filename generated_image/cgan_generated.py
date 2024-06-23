from models.cgan_model import generator 
import torch 
import util.utils as utils
import numpy as np
import os 
import argparse


def GenOptions():
    parser = argparse.ArgumentParser(description='some params for generating image')
    parser.add_argument('--weight', type=str, default='checkpoints/plantvillage_cgan/cganlatest_G.pkl',
                         help='path to the generator weight')
    parser.add_argument('--zdim-in', type=int, default=62, help='number of z channel for input dim')
    parser.add_argument('--input-size', type=int, default=62, help='size of input')
    parser.add_argument('--classes', type=int, default=15, help='number of classes of the dataset')
    parser.add_argument('--zdim-out',type=int, default=3, help='number of classes of the output dim')
    parser.add_argumnet('--batch', type=int, default=225, help='batch size during generation')
    parser.add_argument('--path-save', type='str', default='results', help='path save inference image')
    parser.add_argument('--nimage', type=int, default=50000, help='number of image generated')
    return parser 


def gen_image(model, opt) -> None:
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

if __name__ == '__main__':
    opt = GenOptions().parse()
    
    # check device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load model 
    assert os.path.isfile(opt.weight), f'{opt.weight} is not a valid file'
    state_dict = torch.load(opt.weight)
    model = generator(input_dim=opt.zdim_in, output_dim=opt.zdim_out, 
                      input_size=opt.input_size, class_num=opt.classes)
    model.load_state_dict(state_dict).to(device)
    model.eval()
    
    gen_image(model=model, opt=opt)
