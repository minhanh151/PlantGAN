import torch 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn. functional as F
import torch.optim as optim
import os 
import sys
from torchvision import datasets, transforms
from torch.utils.data import random_split 
from torchvision.utils import save_image

batch_size = 100
learning_rate = 5e-4
max_epoch = 100
device = torch.device("cuda")
num_workers = 5
load_epoch = -1
generate = True
# PATH_ZIP = "/kaggle/input/plantdisease/PlantVillage"
PATH_ZIP = "/home/mia/Downloads/DATA/PlantVillage/train/"
DIR_OUT = "results/vae"
SPLIT_PERCENT = 0.9
BATCH_SIZE = 64

class VAE(nn.Module):
    def __init__(self, shape, latent_size=32):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        c, h, w = shape
        self.hh = (h-1)//2**3 + 1
        self.ww = (w-1)//2**3 + 1

        # For encode
        self.conv_layer = nn.Sequential(
            # (in_channels, out_channels, kernel_size, stride, padding)
            nn.Conv2d(c, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Flatten()
            ) 
        self.fc1 = nn.Sequential(
            nn.Linear(self.hh*self.ww*64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            )   # (batch, 64)
        self.mu = nn.Linear(64, self.latent_size)
        self.logvar = nn.Linear(64, self.latent_size)

        # For decoder
        self.fc3 = nn.Sequential(
            nn.Linear(self.latent_size, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.hh*self.ww*64),
            nn.ReLU()
            )

        self.deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, 2, 1, 1),
            nn.Tanh()
            )

    def encoder(self,x):
        x = self.conv_layer(x)
        x = self.fc1(x)
        mean = self.mu(x)
        logv = self.logvar(x)
        return mean, logv
        

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(device)
        return eps*std + mu
    
    def unFlatten(self, x):
        return x.reshape((x.shape[0], 64, self.hh, self.ww))

    def decoder(self, z):
        t = self.fc3(z)
        t = self.unFlatten(t)
        t = self.deconv_layer(t)
        return t


    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu,logvar)

        # Class conditioning
        pred = self.decoder(z)
        return pred, mu, logvar


def plot(epoch, pred ,name='test_'):
    if not os.path.isdir('./images'):
        os.mkdir('./images')
    
    save_image(pred,"./images/{}epoch_{}.jpg".format(name, epoch), nrow=3)
    plt.close()


def loss_function(x, pred, mu, logvar):
    recon_loss = F.mse_loss(pred, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, kld


def train(epoch, model, train_loader, optim):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    for i,(x,_) in enumerate(train_loader):
        # print(x.shape)
        try:
            optim.zero_grad()   
#             print(x.shape)
            pred, mu, logvar = model(x.to(device))
            
            recon_loss, kld = loss_function(x.to(device),pred, mu, logvar)
            loss = recon_loss + kld
            loss.backward()
            optim.step()

            total_loss += loss.cpu().data.numpy()*x.shape[0]
            reconstruction_loss += recon_loss.cpu().data.numpy()*x.shape[0]
            kld_loss += kld.cpu().data.numpy()*x.shape[0]

        except Exception as e:
            # print("???")
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue
    
    reconstruction_loss /= len(train_loader.dataset)
    kld_loss /= len(train_loader.dataset)
    total_loss /= len(train_loader.dataset)
    return total_loss, kld_loss,reconstruction_loss

def test(epoch, model, test_loader):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    with torch.no_grad():
        for i,(x,y) in enumerate(test_loader):
            try:
                pred, mu, logvar = model(x.to(device))
                recon_loss, kld = loss_function(x.to(device),pred, mu, logvar)
                loss = recon_loss + kld

                total_loss += loss.cpu().data.numpy()*x.shape[0]
                reconstruction_loss += recon_loss.cpu().data.numpy()*x.shape[0]
                kld_loss += kld.cpu().data.numpy()*x.shape[0]
                if i == 0:
                    # print("gr:", x[0,0,:5,:5])
                    # print("pred:", pred[0,0,:5,:5])
                    plot(epoch, pred, y.cpu().data.numpy())
            except Exception as e:
                print("???")
                traceback.print_exc()
                torch.cuda.empty_cache()
#                 continue
    reconstruction_loss /= len(test_loader.dataset)
    kld_loss /= len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    return total_loss, kld_loss,reconstruction_loss        



def generate_image(epoch, z, model):
    with torch.no_grad():
        pred = model.decoder(z.to(device))
        plot(epoch, pred, name='Eval_')
        print("data Plotted")



def load_data():
    transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=transform),batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=transform),batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return train_loader, test_loader

def save_model(model, epoch):
    if not os.path.isdir("./checkpoints"):
        os.mkdir("./checkpoints")
    file_name = './checkpoints/model_{}.pt'.format(epoch)
    torch.save(model.state_dict(), file_name)



if __name__ == "__main__":
    
    print("dataloader created")
    model = VAE((3,64,64)).to(device)
    print("model created")
    data_train = datasets.ImageFolder(PATH_ZIP, transform=transforms.Compose([
                                 transforms.Resize(64),
                                 transforms.ToTensor(),
                                #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))
    num_train = int(len(data_train) * SPLIT_PERCENT)
    data_train, data_valid = random_split(data_train, [num_train, len(data_train) - num_train])
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_valid, batch_size=BATCH_SIZE, shuffle=False)
    if load_epoch > 0:
        model.load_state_dict(torch.load('./checkpoints/model_{}.pt'.format(load_epoch), map_location=torch.device('cpu')))
        print("model {} loaded".format(load_epoch))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)


    train_loss_list = []
    test_loss_list = []
    for i in range(load_epoch+1, max_epoch):
        model.train()
        train_total, train_kld, train_loss = train(i, model, train_loader, optimizer)
        with torch.no_grad():
            model.eval()
            test_total, test_kld, test_loss = test(i, model, test_loader)
            if generate:
                z = torch.randn(15, 32).to(device)
                generate_image(i,z, model)
            
        print("Epoch: {}/{} Train loss: {}, Train KLD: {}, Train Reconstruction Loss:{}".format(i, max_epoch,train_total, train_kld, train_loss))
        print("Epoch: {}/{} Test loss: {}, Test KLD: {}, Test Reconstruction Loss:{}".format(i, max_epoch, test_loss, test_kld, test_loss))

        save_model(model, i)
        train_loss_list.append([train_total, train_kld, train_loss])
        test_loss_list.append([test_total, test_kld, test_loss])
        np.save("train_loss", np.array(train_loss_list))
        np.save("test_loss", np.array(test_loss_list))


    # i, (example_data, exaple_target) = next(enumerate(test_loader))
    # print(example_data[0,0].shape)
    # plt.figure(figsize=(5,5), dpi=100)
    # plt.imsave("example.jpg", example_data[0,0], cmap='gray',  dpi=1000)
    