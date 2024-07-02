import torch 
import torchvision

import numpy as np
import time, os, pickle
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn. functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import random_split 
from torchvision.utils import save_image


from .base_model import BaseModel
from . import networks
import util.utils as utils
from util.visualizer import Visualizer


class Encoder(nn.Module):
    '''
    Simple Encoder for VAE model
    '''
    def __init__(self, input_dim=3, input_size=64, latent_size=32, num_classes=15):
        super(Encoder,self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes 
        self.input_size = input_size
        self.input_mlp = (self.input_size - 1)//2**3 + 1 # input size for fc layer
         
        self.conv_layer = nn.Sequential(
            nn.Conv2d(input_dim, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Flatten()
            )

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_mlp*self.input_mlp*64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ) 

        self.mu = nn.Linear(64, self.latent_size)
        self.logvar = nn.Linear(64, self.latent_size)
        
    def forward(self, t):
        t = self.conv_layer(t)
        # print(t.shape)
        t = self.fc1(t)
        mean = self.mu(t)
        logv = self.logvar(t)
        return mean, logv

class Decoder(nn.Module):
    '''
    Simple Decoder for VAE model
    '''
    def __init__(self, input_size=64, latent_size=32, num_classes=15):
        super(Decoder,self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes 
        self.input_size = input_size
        self.input_mlp = (self.input_size - 1)//2**3 + 1 # input size for fc layer
        
        self.fc3 = nn.Sequential(
            nn.Linear(self.latent_size + self.num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 64*self.input_mlp*self.input_mlp),
            nn.ReLU()
            )

        self.deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, 2, 1, 1),
            nn.Tanh()
            )
    
    def unFlatten(self, x):
        return x.reshape((x.shape[0], 64, self.input_mlp, self.input_mlp))
    
    def forward(self, z):
        t = self.fc3(z)
        t = self.unFlatten(t)
        t = self.deconv_layer(t)
        return t 


class CVAE(nn.Module):
    def __init__(self, input_dim=3, input_size=64, latent_size=32, num_classes=15):
        '''
        Simple Conditional VAE
        '''
        super(CVAE,self).__init__()
        self.encoder = Encoder(input_dim, input_size, latent_size, num_classes)
        self.decoder = Decoder(input_size, latent_size, num_classes)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps*std + mu

    def forward(self, x, y):
        # change shape of y to the input x shape 
        y_input = torch.argmax(y, dim=1).reshape((y.shape[0],1,1,1))
        y_input = torch.ones(x.shape, device=x.device)*y_input
        t = torch.cat((x,y_input),dim=1)
        
        mu, logvar = self.encoder(t)
        z = self.reparameterize(mu,logvar)
        
        z = torch.cat((z, y.float()), dim=1)
        pred = self.decoder(z)
        
        return pred, mu, logvar
    
class CVAEModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        self.epoch = opt.epoch_count
        self.z_dim = 6
        self.laten_size = 32
        # self.batch_size = opt.batch_size
        self.batch_size = opt.batch_size
        self.input_size = opt.input_size
        self.sample_num = self.opt.class_num ** 2
        self.dataset = opt.name
        self.result_dir = opt.result_dir
        self.log_dir = opt.log_dir
        self.model_name = opt.model
        
        
        
        transform = transforms.Compose([transforms.Resize((opt.input_size, opt.input_size)), 
                                        transforms.ToTensor(), 
                                        # transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                                            #   std=(0.5, 0.5, 0.5)) 
                                        ])
        
        data_all = datasets.ImageFolder(opt.dataroot, transform=transform)
        num_train = int(len(data_all) * 0.9)
        data_train, data_valid = random_split(data_all, [num_train, len(data_all) - num_train])        
        
        self.netData_loader = torch.utils.data.DataLoader(
            data_train,
            batch_size= opt.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(data_valid, batch_size=opt.batch_size, shuffle=False)
        
        dataset = self.netData_loader.__iter__().__next__()[0]
        self.model = CVAE(input_dim=self.z_dim, input_size=self.input_size,
                        latent_size=self.laten_size, num_classes=self.opt.class_num)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=0.001)

        if self.opt.gpu_mode:
            self.model.cuda()
        
        print('---------- Networks architecture -------------')
        utils.print_network(self.model)
        print('----------------------------------------------')
    
    
    def train(self):
        print('training start!!')
        start_time = time.time()
        total_iters = 0                # the total number of training iterations
        # visualizer = Visualizer(self.opt)   # create a visualizer that display/save images and plots
        dataset_size = self.netData_loader.dataset.__len__()

        self.train_hist = {}
        self.train_hist['kld_loss'] = []
        self.train_hist['recon_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_loss'] = []
        self.train_hist['total_time'] = []
        
        
        for epoch in range(int(self.epoch)):
            epoch_start_time = time
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0 
            self.model.train()
            
            
            for iter,(x,y) in enumerate(self.netData_loader):
                reconstruction_loss = 0
                kld_loss = 0
                total_loss = 0
                iter_start_time = time.time()
        
                if total_iters % self.opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                
                # visualizer.reset()
                total_iters += self.opt.batch_size
                epoch_iter += self.opt.batch_size
                
                if iter == self.netData_loader.dataset.__len__() // self.batch_size:
                    break

                label = np.zeros((x.shape[0], 15))
                label[np.arange(x.shape[0]), y] = 1
                label = torch.tensor(label)
                
                if self.opt.gpu_mode:
                    x, label = x.cuda(), label.cuda()
                self.optimizer.zero_grad()   
                pred, mu, logvar = self.model(x, label)
                recon_loss, kld = loss_function(x,pred, mu, logvar)
                loss = recon_loss + kld
                
                self.train_hist['kld_loss'].append(kld)
                self.train_hist['recon_loss'].append(recon_loss)
                self.train_hist['total_loss'].append(loss)
                
                loss.backward()
                self.optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] kld_loss: %.8f, recon_loss: %.8f, total_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.netData_loader.dataset.__len__() // self.batch_size, 
                           kld, recon_loss, loss))
                
                total_loss += loss.cpu().data.numpy()*x.shape[0]
                reconstruction_loss += recon_loss.cpu().data.numpy()*x.shape[0]
                kld_loss += kld.cpu().data.numpy()*x.shape[0]
                
                if total_iters % self.opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % self.opt.update_html_freq == 0
                    self.compute_visuals()
                    # visualizer.display_current_results(self.get_current_visuals(), epoch, save_result)
                
                if total_iters % self.opt.print_freq == 0:    # print training losses and save logging information to the disk
                    # losses = self.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / self.opt.batch_size
                    # visualizer.print_current_losses(epoch, epoch_iter, total_loss, t_comp, t_data)
                    # if self.opt.display_id > 0:
                        # visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, total_loss)
                
                iter_data_time = time.time()
            
            # break
            if epoch % self.opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                self.save(epoch)
            
            reconstruction_loss /= len(self.netData_loader.dataset)
            kld_loss /= len(self.netData_loader.dataset)
            total_loss /= len(self.netData_loader.dataset)

            print('End of epoch %d \t Time Taken: %d sec' % (epoch, time.time() - epoch_start_time))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.generate_image(epoch)
                # self.visualize_results((epoch+1))
            
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
            self.epoch, self.train_hist['total_time'][0]))   
    
    def generate_image(self, epoch):
        z = torch.randn(15, 32)
        y = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) - 1
        label = np.zeros((y.shape[0], 15))
        label[np.arange(z.shape[0]), y] = 1
        label = torch.tensor(label)
        if self.opt.gpu_mode:
             z, label = z.cuda(), label.cuda()
        pred = model.decoder(torch.cat((z,label.float()), dim=1))
        plot(epoch, pred, y.cpu().data.numpy(),name='Eval_')
        print("data Plotted")

    def save(self, epoch):

        save_dir = os.path.join(self.save_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.model.state_dict(), os.path.join(save_dir, self.model_name + str(epoch) + '.pkl'))
        # torch.save(self.netD.state_dict(), os.path.join(save_dir, self.model_name + str(epoch) + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + str(self.epoch) + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)
            

            
def plot(epoch, pred, y,name='test_'):
    if not os.path.isdir('./images'):
        os.mkdir('./images')
    
    save_image(pred,"./images/{}epoch_{}.jpg".format(name, epoch), nrow=3)
    plt.close()


def loss_function(x, pred, mu, logvar):
    recon_loss = F.mse_loss(pred, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, kld



def test(epoch, model, test_loader):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    with torch.no_grad():
        for i,(x,y) in enumerate(test_loader):
            try:
                label = np.zeros((x.shape[0], 15))
                label[np.arange(x.shape[0]), y] = 1
                label = torch.tensor(label)

                pred, mu, logvar = model(x.to(device),label.to(device))
                recon_loss, kld = loss_function(x.to(device),pred, mu, logvar)
                loss = recon_loss + kld

                total_loss += loss.cpu().data.numpy()*x.shape[0]
                reconstruction_loss += recon_loss.cpu().data.numpy()*x.shape[0]
                kld_loss += kld.cpu().data.numpy()*x.shape[0]
                if i == 0:
                    plot(epoch, pred, y.cpu().data.numpy())
            except Exception as e:
                traceback.print_exc()
                torch.cuda.empty_cache()
#                 continue
    reconstruction_loss /= len(test_loader.dataset)
    kld_loss /= len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    return total_loss, kld_loss,reconstruction_loss        


def save_model(model, epoch):
    if not os.path.isdir("./checkpoints"):
        os.mkdir("./checkpoints")
    file_name = './checkpoints/model_{}.pt'.format(epoch)
    torch.save(model.state_dict(), file_name)
 

    