import torch
import torch.nn as nn

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)

class generator(nn.Module):
    # initializers
    def __init__(self, batch_size1):
        super(generator, self).__init__()
        self.batch_size1 = batch_size1
        self.latent = nn.Sequential(
            nn.Linear(100, 512 * 4 * 4),
            nn.LeakyReLU(0.05)
        )
        self.label_latent = nn.Sequential(
            nn.Linear(10, 100),
            nn.Linear(100, 4 * 4)
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
        x = torch.cat([self.latent(input).reshape((self.batch_size1, 512, 4, 4))[0:self.size_batch_mid],
                       self.label_latent(label).reshape((self.size_batch_mid, 1, 4, 4))], 1)
        x = self.initial_block(x)
        x = self.residual_block(x)
        x = self.conv_B(x)
        return x


class discriminator(nn.Module):
    def __init__(self, resblock, batch_size1, device1):
        super(discriminator, self).__init__()
        self.device1 = device1
        self.initial = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False)
        )
        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False),
            resblock(512, 512, downsample=False),
        )
        self.label_latent = nn.Sequential(
            nn.Linear(10, 100),
            nn.Linear(100, 128 * 128 * 1)
        )

    def forward(self, input, label):
        self.size_label_batch = self.label_latent(label).size()[0]
        x = self.label_latent(label).reshape((self.size_label_batch, 1, 128, 128))
        x = torch.cat([input[0:self.size_label_batch], x], 1)
        input = self.initial(x)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input)
        output = input.size()[0]
        input = torch.nn.Linear(output, 1).to(self.device1)(input)

        return nn.Sigmoid()(input)
