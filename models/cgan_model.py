import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import util.utils as utils
import time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from util.visualizer import Visualizer
from torchvision import datasets, transforms




class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32, class_num=10):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.netDeconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.netDeconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32, class_num=10):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim + self.class_num, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x



class CGANModel(BaseModel):
    """
    This class implements the CGAN model

    The model training requires '--dataset_mode unaligned' dataset.

    CGAN paper: 
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')

        return parser

    def __init__(self, opt):
        """Initialize the CGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # parameters
        self.epoch = opt.epoch_count
        self.z_dim = 62
        self.batch_size = opt.batch_size
        self.input_size = opt.input_size
        self.sample_num = self.opt.class_num ** 2
        self.dataset = opt.name
        self.result_dir = opt.result_dir
        self.log_dir = opt.log_dir
        self.model_name = opt.model

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D', 'G']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['G_']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        transform = transforms.Compose([transforms.Resize((opt.input_size, opt.input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        self.netData_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(opt.dataroot, transform=transform),
            batch_size= opt.batch_size, shuffle=True)
        dataset = self.netData_loader.__iter__().__next__()[0]

        # define networks (both Generators and discriminators)
        self.netG = generator(input_dim=self.z_dim, output_dim=dataset.shape[1], input_size=self.opt.input_size, class_num=self.opt.class_num)
        self.netD = discriminator(input_dim=dataset.shape[1], output_dim=1, input_size=self.opt.input_size, class_num=self.opt.class_num)
        self.netG_optimizer = optim.Adam(self.netG.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
        self.netD_optimizer = optim.Adam(self.netD.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))

        if self.opt.gpu_mode:
            self.netG.cuda()
            self.netD.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.netG)
        utils.print_network(self.netD)
        print('-----------------------------------------------')

        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.opt.class_num):
            self.sample_z_[i*self.opt.class_num] = torch.rand(1, self.z_dim)
            for j in range(1, self.opt.class_num):
                self.sample_z_[i*self.opt.class_num + j] = self.sample_z_[i*self.opt.class_num]

        temp = torch.zeros((self.opt.class_num, 1))
        for i in range(self.opt.class_num):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.opt.class_num):
            temp_y[i*self.opt.class_num: (i+1)*self.opt.class_num] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.opt.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        if self.opt.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

    
    def train(self):

        print('training start!!')
        start_time = time.time()
        total_iters = 0                # the total number of training iterations
        visualizer = Visualizer(self.opt)   # create a visualizer that display/save images and plots
        dataset_size = self.netData_loader.dataset.__len__()

        self.train_hist = {}
        self.train_hist['loss_D'] = []
        self.train_hist['loss_G'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.opt.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.netD.train()

        for epoch in range(int(self.epoch)):
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

            self.netG.train()

            for iter, (x_, y_) in enumerate(self.netData_loader):
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % self.opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                visualizer.reset()
                total_iters += self.opt.batch_size
                epoch_iter += self.opt.batch_size

                if iter == self.netData_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))
                y_vec_ = torch.zeros((self.batch_size, self.opt.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
                y_fill_ = y_vec_.unsqueeze(2).unsqueeze(3).expand(self.batch_size, self.opt.class_num, self.input_size, self.input_size)
                if self.opt.gpu_mode:
                    x_, z_, y_vec_, y_fill_ = x_.cuda(), z_.cuda(), y_vec_.cuda(), y_fill_.cuda()

                # update D network
                self.netD_optimizer.zero_grad()

                D_real = self.netD(x_, y_fill_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                self.G_ = self.netG(z_, y_vec_)
                D_fake = self.netD(self.G_, y_fill_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                self.loss_D = D_real_loss + D_fake_loss
                self.train_hist['loss_D'].append(self.loss_D.item())

                self.loss_D.backward()
                self.netD_optimizer.step()

                # update G network
                self.netG_optimizer.zero_grad()

                self.G_ = self.netG(z_, y_vec_)
                D_fake = self.netD(self.G_, y_fill_)
                self.loss_G = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['loss_G'].append(self.loss_G.item())

                self.loss_G.backward()
                self.netG_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] loss_D: %.8f, loss_G: %.8f" %
                          ((epoch + 1), (iter + 1), self.netData_loader.dataset.__len__() // self.batch_size, self.loss_D.item(), self.loss_G.item()))

                # break
                if total_iters % self.opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % self.opt.update_html_freq == 0
                    self.compute_visuals()
                    visualizer.display_current_results(self.get_current_visuals(), epoch, save_result)

                if total_iters % self.opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = self.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / self.opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if self.opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                if total_iters % self.opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if self.opt.save_by_iter else 'latest'
                    self.save(save_suffix)

                iter_data_time = time.time()

            # break
            if epoch % self.opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                self.save(epoch)
                # self.save_networks(epoch)

            print('End of epoch %d \t Time Taken: %d sec' % (epoch, time.time() - epoch_start_time))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        # self.save()
        utils.generate_animation(self.result_dir + '/' + self.model_name + '/' + self.dataset + '/' + self.model_name,
                                 self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir), self.model_name)

  
    def visualize_results(self, epoch, fix=True):
        self.netG.eval()

        if not os.path.exists(self.result_dir + '/' + self.model_name + '/'+ self.dataset):
            os.makedirs(self.result_dir + '/' + self.model_name + '/' + self.dataset)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        if fix:
            """ fixed noise """
            samples = self.netG(self.sample_z_, self.sample_y_)
        else:
            """ random noise """
            sample_y_ = torch.zeros(self.batch_size, self.opt.class_num).scatter_(1, torch.randint(0, self.opt.class_num - 1, (self.batch_size, 1)).type(torch.LongTensor), 1)
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.opt.gpu_mode:
                sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()

            samples = self.netG(sample_z_, sample_y_)

        if self.opt.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.model_name + '/' + self.dataset  +'/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self, epoch):

        save_dir = os.path.join(self.save_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.netG.state_dict(), os.path.join(save_dir, self.model_name + str(epoch) + '_G.pkl'))
        torch.save(self.netD.state_dict(), os.path.join(save_dir, self.model_name + str(epoch) + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + str(self.epoch) + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights