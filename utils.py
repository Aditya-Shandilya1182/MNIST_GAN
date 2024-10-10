import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision

def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available():
        return n.cuda()
    return n

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)

def save_generated_images(images, epoch):
    images = images * 0.5 + 0.5
    grid = torchvision.utils.make_grid(images, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"Epoch {epoch}")
    plt.savefig(f'gan_epoch_{epoch}.png')
    plt.show()
