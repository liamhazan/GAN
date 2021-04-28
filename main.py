import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
from torchvision import transforms
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils
from models import Discriminator, Generator
from random import choices, random


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.50612344, 0.42543389, 0.38283129)
                                             ,(0.31063245, 0.29027997, 0.28964681))])

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def gen_z(batch_size, discrete_latent_dim, categorical_dim, continuous_latent_dim):
    cont_z = torch.randn(batch_size,continuous_latent_dim)
    rand_places = torch.tensor(choices(range(categorical_dim), k=batch_size*discrete_latent_dim)).reshape(batch_size,-1)
    disc_z = torch.zeros(batch_size,discrete_latent_dim,categorical_dim)
    for i in range(batch_size):
        disc_z[i,torch.arange(discrete_latent_dim),rand_places[i]] = 1
    return torch.cat([cont_z,disc_z.reshape(batch_size,-1)],dim=1).reshape(batch_size,categorical_dim*discrete_latent_dim+continuous_latent_dim,1,1)



train_dataset = torchvision.datasets.ImageFolder(root="./all_images",transform=transform)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=120, shuffle=True)


discrete_latent_dim = 10
categorical_dim =  2
continuous_latent_dim = 10
len_z = discrete_latent_dim*categorical_dim + continuous_latent_dim
EPOCHS = 1500
real_label = 1
fake_label = 0

discriminator = Discriminator()
generator = Generator(len_z)
if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()


criterion = nn.BCELoss()


d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003)
res_dict = {"discriminator_loss":[], "generator_loss":[]}

for epoch in range(EPOCHS):
    print(f"start epoch: {epoch}")
    accuracy_real = 0
    accuracy_fake = 0
    d_total_loss = 0
    g_total_loss = 0
    for (images, _) in tqdm(train_dataloader):
        batch_size = images.size(0)
        images = to_cuda(images)

        real_labels = to_cuda(torch.FloatTensor(batch_size).uniform_(0.9, 1))
        fake_labels = to_cuda(torch.FloatTensor(batch_size).uniform_(0, 0.1))
        if random() < 0.03:
            fake_labels = to_cuda(torch.FloatTensor(batch_size).uniform_(0.9, 1))
            real_labels = to_cuda(torch.FloatTensor(batch_size).uniform_(0, 0.1))


        # ============= Train the discriminator =============#
        discriminator.zero_grad()
        outputs = discriminator(images.detach()).reshape(-1)  # Real images
        d_loss_real = criterion(outputs, real_labels)


        z = to_cuda(gen_z(batch_size,discrete_latent_dim,categorical_dim,continuous_latent_dim))
        fake_images = generator(z)  # Generate fake images
        outputs = discriminator(fake_images.detach()).reshape(-1)
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_total_loss += d_loss.item()
        d_loss.backward()
        d_optimizer.step()

        # if random() < 0.1:
        #     continue
        # =============== Train the generator ===============#
        generator.zero_grad()
        real_labels = to_cuda(torch.ones(batch_size))
        outputs = discriminator(fake_images).reshape(-1)
        g_loss = criterion(outputs, real_labels)
        g_total_loss += g_loss.item()

        g_loss.backward()
        g_optimizer.step()


    if (epoch+1)%20 == 0:
        with torch.no_grad():
            z = to_cuda(gen_z(32, discrete_latent_dim, categorical_dim, continuous_latent_dim))
            fake_images = generator(z)
            grid_fake = torchvision.utils.make_grid(fake_images[:32], normalize=True)
            plt.imshow(np.transpose(grid_fake.cpu().detach().numpy(), (1,2,0)))
            plt.savefig(f"./visualizations/grid_fake_epoch_{epoch+1}.png")
            plt.clf()
            if torch.cuda.is_available():
                torch.save(discriminator.state_dict(), f"discriminator_epoch:{epoch}.pkl")
                torch.save(generator.state_dict(), f"generator_epoch:{epoch}.pkl")

    res_dict["discriminator_loss"].append(d_total_loss/len(train_dataset))
    res_dict["generator_loss"].append(g_total_loss/len(train_dataset))


    plt.plot(res_dict["discriminator_loss"], c="red", label="Discriminator loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()

    plt.plot(res_dict["generator_loss"], c="blue", label="Generator loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig('./visualizations/loss-epochs.png')
    plt.clf()


