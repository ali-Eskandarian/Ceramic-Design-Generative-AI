import numpy as np
import pandas as pd
import random
import os, time
import matplotlib.pyplot as plt
import itertools
import pickle
import cv2
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision

fixed_noise = np.random.randn(10,200,100)

cuda = torch.device('cuda')     # Default CUDA device

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class generator(nn.Module):
    # initializers
    def __init__(self, batch_size1):
        super(generator, self).__init__()
        self.batch_size1 =batch_size1
        self.latent=nn.Sequential(
            nn.Linear(100, 512*4*4),
            nn.LeakyReLU(0.05)
        )
        self.label_latent = nn.Sequential(
            nn.Linear(10,100),
            nn.Linear(100,4*4)
        )
        self.initial_block = nn.Sequential(
            nn.ConvTranspose2d(513, 256, kernel_size=3, stride=3, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.residual_block = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv_B = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        # print(self.latent(input).size())
        self.size_batch_mid = self.label_latent(label).size()[0]
        x = torch.cat([self.latent(input).reshape((self.batch_size1, 512,4,4))[0:self.size_batch_mid], self.label_latent(label).reshape((self.size_batch_mid, 1,4,4))], 1)
        x = self.initial_block(x)
        x = self.residual_block(x)
        x = self.conv_B(x)
        return x

images = np.zeros((10,200,3,128,128))

for p in range(10):
  images[p] =np.array([np.array(torchvision.io.read_image(rf'C:\Users\ali\PycharmProjects\ACGAN-Kashi\selecting\{p}_{i}.jpg').float()) for i in range(200)])
  print(p)

G = generator(50).cuda()
G.load_state_dict(torch.load(r'C:\Users\ali\PycharmProjects\ACGAN-Kashi\GAN_kashi_Eskandarian\generator.pt'))

# Validation using MSE Loss function
loss_function = torch.nn.CrossEntropyLoss()
 
# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(G.parameters(),
                             lr = 1e-1)
epochs = 200
outputs = []
loss_ep=[]
label1 = torch.from_numpy(np.array([1]*50))
l = F.one_hot(label1,10).float().cuda()

for epoch in range(epochs):
    lst = np.linspace(0,200,200,endpoint = False,dtype = int)
    np.random.shuffle(lst)
    lst = lst.reshape(40,5)
    data_noise = np.zeros((40,50,100))
    data_images = np.zeros((40,50,3,128,128))
    for batching in range(40):
      indexes = lst[batching]
      data_noise[batching] = fixed_noise[:,indexes,:].reshape(50,100)
      data_images[batching] = images[:,indexes,:,:,:].reshape(50,3,128,128)
    t = torch.from_numpy(data_noise[:,:,:]).float().cuda()
    x = torch.from_numpy(data_images[:,:,:,:,:]).float().cuda()
    print(f"epoch :  {epoch}")
    losses = 0
    for i in range(40):
      image, fixed_noise_img = x[i]/255, t[i]

      reconstructed = G(fixed_noise_img, l)

      loss = loss_function(reconstructed, image)
      # print(loss)
      optimizer.zero_grad()
      loss.backward() 
      optimizer.step()

      losses+=loss/40
    loss_ep.append(losses)
    outputs.append((epochs, image, reconstructed))
    print(f"loss mean  :  {loss_ep[epoch]}")
    # if epoch>5:
    #   if (((loss_ep[-1]-loss_ep[-2])/loss_ep[-2])<0.001) and (((loss_ep[-2]-loss_ep[-3])/loss_ep[-3])<0.001) and (((loss_ep[-3]-loss_ep[-4])/loss_ep[-4])<0.001) :
    #     torch.save(G.state_dict(), '/content/drive/MyDrive/kashi/generator.pt')
    #     break
    if epoch>50:
      torch.save(G.state_dict(), r'C:\Users\ali\PycharmProjects\ACGAN-Kashi\generator.pt')
# # Plotting the last 100 values
for n, i in enumerate(loss_ep):
  loss_ep[n] = i.detach().cpu().numpy()
# # Defining the Plot Style
plt.plot(loss_ep)
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')


# In[16]:




# In[40]:


images_gen = G(fixed_noise_img,l).detach().cpu().numpy()


# In[41]:


images_gen.shape


# In[49]:


np.array(images_gen[0].T*255,dtype='uint8')


# In[51]:


for image in images_gen:
  plt.imshow(np.array(image.T*255,dtype='uint8'))
  plt.show()


# In[ ]:




