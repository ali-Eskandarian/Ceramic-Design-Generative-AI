from matplotlib import image
import numpy as np
from PIL import Image
import PIL
import keras
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob
import random
from copy import deepcopy as dc
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from model import generator, discriminator
from dataset import ImageDataset
from torch.utils.data import DataLoader


data = ImageDataset("labels.csv","saved/")
num_classes = 10
batch_size = 16

transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_dataloader =DataLoader(data, batch_size=batch_size, shuffle=True)

G = generator(batch_size1=batch_size)
D = discriminator(batch_size1=batch_size)
G.cuda()
D.cuda()
# G = torch.load(r'C:\Users\ali\PycharmProjects\ACGAN-Kashi\generator.pt')
G.load_state_dict(torch.load(r'C:\Users\ali\PycharmProjects\ACGAN-Kashi\generator.pt'))
D.weight_init(mean=0, std=0.02)

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

G.train()
D.train()

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 10

fixed_noise= torch.randn(batch_size, NOISE_DIM).to(device)
n = 0
#
for epoch in range(NUM_EPOCHS):
        for image_real, label_data1 in (train_dataloader):
                n+=1
                label_data = F.one_hot(label_data1,10).float().to(device)
                real = image_real.to(device)
                noise = torch.randn(batch_size, NOISE_DIM).to(device)
                fake = G(noise,label_data)

                ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                D.zero_grad()
                disc_real = D(real,label_data).reshape(-1)
                loss_disc_real = BCE_loss(disc_real, torch.ones_like(disc_real))
                disc_fake = D(fake.detach(),label_data).reshape(-1)
                loss_disc_fake = BCE_loss(disc_fake, torch.zeros_like(disc_fake))
                loss_disc = (loss_disc_real + loss_disc_fake) / 2
                loss_disc.backward()
                D_optimizer.step()

                ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                G.zero_grad()
                output = D(fake, label_data).reshape(-1)
                loss_gen = BCE_loss(output, torch.ones_like(output))
                loss_gen.backward()
                G_optimizer.step()

                # Print losses occasionally and print to tensorboard
                if n % 100 == 0:
                        print(
                                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {(n-epoch*4105)}/{len(train_dataloader)} \
                      Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                        )
                if n%200==0 and (fixed_noise.size()[0]==label_data.size()[0]):

                        with torch.no_grad():

                                # label_data = F.one_hot(label_data, 10).float().to(device)
                                fake11 = G(fixed_noise,label_data)
                                fake11 = fake11.cpu().numpy()[0,:,:,:].T
                                filename0 = fr"C:\Users\ali\PycharmProjects\ACGAN-Kashi\generated\{n}.jpg"
                                im1 = Image.fromarray(np.uint8(255* fake11))
                                im1 = im1.save(filename0)
# G.load_state_dict(torch.load(r"C:\Users\ali\PycharmProjects\ACGAN-Kashi\gen.pth"))
# D.load_state_dict(torch.load(r"C:\Users\ali\PycharmProjects\ACGAN-Kashi\Disc.pth"))
# a = torch.load(G.state_dict(), r"C:\Users\ali\PycharmProjects\ACGAN-Kashi\gen.pth")
# b = torch.load(D.state_dict(), r"C:\Users\ali\PycharmProjects\ACGAN-Kashi\disc.pth")
# print(a , b)
label_data = F.one_hot(torch.arange(0, batch_size) % 1).float().to(device)
plt.imshow(G(fixed_noise,label_data))
plt.show()

# for i in range(10):
#         label_data = F.one_hot(i, 10).float().to(device)
#         noise = torch.randn(batch_size, NOISE_DIM).to(device)
#         fake = G(noise, label_data)
#         print(fake.size())
#         plt.imshow(fake)
#         plt.show()