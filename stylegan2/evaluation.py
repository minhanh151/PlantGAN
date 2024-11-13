import sys
# print(sys.path)
sys.path.append('/home/mia/Downloads/GitHub/PlantGAN')
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
from scipy.stats import entropy

import torchvision.transforms as transforms
from PIL import Image 
import os

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, (batch, _) in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def fid_score(real_dataset, fake_dataset, device, batch_size=32, num_feature=2048):
    # create data loader
    real_data_loader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size)
    generated_data_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=batch_size)

    # Calculate FID Score
    fid = FrechetInceptionDistance(feature=num_feature).to(device)  # Use 2048 features (default)

    for real_images, _ in real_data_loader:
        fid.update(real_images.type(torch.uint8).to(device), real=True)  # Update with real images

    for generated_images, _ in generated_data_loader:
        fid.update(generated_images.type(torch.uint8).to(device), real=False)  # Update with generated images

    fid_score = fid.compute()
    return fid_score

class TrueDataset(torch.utils.data.Dataset):
        def __init__(self, dataset_path, transform=transforms.Compose([
                                    # transforms.Resize(32),
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])):
            self.list_img = []
            image_extensions = {".jpg", ".jpeg", ".png"}
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    # only take damage leaf
                    if 'healthy' in root.split('/')[-1]:
                        continue
                    
                    if os.path.splitext(file)[1].lower() in image_extensions:
                        self.list_img.append(os.path.join(root, file))

            self.transform = transform
            
        def __getitem__(self, index):
            img_path = self.list_img[index]
            img = Image.open(img_path)
            img = np.array(img)
            img = img.transpose(2, 0, 1) # HWC => CHW
            # if self.transform:
                # img = self.transform(img)
            
            return img, []
        def __len__(self):
            return len(self.list_img)
        
if __name__ == '__main__':

    
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fake_dataset = TrueDataset(dataset_path='/home/mia/Downloads/GitHub/PlantGAN/results/gen_sb_ckpt2500_lpips_v2-1')
    true_dataset = TrueDataset(dataset_path="/home/mia/Downloads/DATA/PlantVillage_stylegan")
    
    print(inception_score(fake_dataset, cuda=True, batch_size=32, resize=True, splits=10))
    print(fid_score(true_dataset, fake_dataset, device=device)) 