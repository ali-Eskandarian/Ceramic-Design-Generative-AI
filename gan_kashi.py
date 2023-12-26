import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import  torch.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import generator, discriminator, ResBlock
from dataset import ImageDataset
from PIL import Image

#writer
writer = SummaryWriter("runs/kashi")

# create dataset
Labels_address = r"C:\Users\ali\PycharmProjects\ACGAN-Kashi\selecting_labels.csv"
Images_address = r"C:\Users\ali\PycharmProjects\ACGAN-Kashi\selecting"
data = ImageDataset(Labels_address, Images_address)

# Hyper parameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 1000
num_classes = 10
batch_size2 = 16
best_gan_model = np.inf
G_loss = []
D_loss = []

# creating dataloader
train_dataloader = DataLoader(data, batch_size=batch_size2, shuffle=True)

# initial Models and load last states
G = generator(batch_size1=batch_size2).cuda()
D = discriminator(ResBlock, batch_size2, device).cuda()
G.load_state_dict(torch.load(r'C:\Users\ali\PycharmProjects\ACGAN-Kashi\generator_epoch.pt'))
D.load_state_dict(torch.load(r'C:\Users\ali\PycharmProjects\ACGAN-Kashi\discriminator_epoch.pt'))

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas = (0.5, 0.99))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas = (0.5, 0.99))

# train mode for models
G.train()
D.train()

# create fixed noise
fixed_noise = torch.randn(batch_size2, NOISE_DIM).to(device)

for epoch in range(NUM_EPOCHS):

    loss_g_mean = 0
    loss_d_mean = 0
    for image_real, label_data1 in train_dataloader:
        label_data1 = F.one_hot(label_data1, 10).float().to(device)
        real = image_real.to(device)
        noise = torch.randn(batch_size2, NOISE_DIM).to(device)
        fake = G(noise, label_data1)
        writer.add_graph(G, (fixed_noise,label_data1) )

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        D.zero_grad()
        disc_real = D(real, label_data1).reshape(-1)
        loss_disc_real = BCE_loss(disc_real, torch.ones_like(disc_real))
        disc_fake = D(fake.detach(), label_data1).reshape(-1)
        loss_disc_fake = BCE_loss(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        loss_disc.backward()
        D_optimizer.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        G.zero_grad()
        output = D(fake, label_data1).reshape(-1)
        loss_gen = BCE_loss(output, torch.ones_like(output))
        loss_gen.backward()
        G_optimizer.step()
        loss_g_mean += loss_gen.detach().cpu().numpy() / batch_size2
        loss_d_mean += loss_disc.detach().cpu().numpy() / batch_size2

    with torch.no_grad():
        try:
            images_gen = G(fixed_noise, label_data1).detach().cpu().numpy()
            for lp, image in enumerate(images_gen):
                filename0 = fr"C:\Users\ali\PycharmProjects\ACGAN-Kashi\generated\{epoch}_{lp}.jpg"
                im1 = Image.fromarray(np.array(image.T * 255, dtype='uint8'))
                writer.add_image("fake", vutils.make_grid(im1), lp)
                # im1.save(filename0)
            writer.close()
            sys.exit()
        except:
            print("Can't save")

    print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {len(train_dataloader)} \
                Loss D: {loss_d_mean:.4f}, loss G: {loss_g_mean:.4f}, best_G_loss: {best_gan_model} ")

    G_loss.append(loss_g_mean)
    D_loss.append(loss_d_mean)
    torch.save(G.state_dict(), r"C:\Users\ali\PycharmProjects\ACGAN-Kashi\models\disc\generator_epoch.pt")
    torch.save(D.state_dict(), r"C:\Users\ali\PycharmProjects\ACGAN-Kashi\models\disc\discriminator_epoch.pt")

plt.plot(G_loss)
plt.show()
plt.plot(D_loss)
plt.show()
