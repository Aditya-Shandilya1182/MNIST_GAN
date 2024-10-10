import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from generative_net import GenerativeNet
from discriminative_net import DiscriminativeNet
from train import train_gan
from utils import init_weights

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 64
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

generator = GenerativeNet()
discriminator = DiscriminativeNet()
generator.apply(init_weights)
discriminator.apply(init_weights)

train_gan(generator, discriminator, train_loader, num_epochs=25)
