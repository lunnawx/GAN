#%%
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torchvision.utils as vutils

image_size = 64
batch_size = 1024
dataroot = "CelebA/img_align_celeba"
dataset = torchvision.datasets.ImageFolder(root=dataroot, 
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
    ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#%%
real_batch = next(iter(dataloader))


# %%
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
# %%
nz = 100
ngf = 64
nc = 3 
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),

        )
    def forward(self, x):
        return self.main(x)
G = Generator().to(device)
G.apply(weight_init)
#%%
ndf = 64
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1,bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1,bias=True),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1,bias=True),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1,bias=True),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x)
D = Discriminator().to(device)
D.apply(weight_init)
print(D)
# %%
loss_fn = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=[0.5,0.999])
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=[0.5,0.999])

#%%
total_step = len(dataloader)
num_epochs = 5
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
         D.zero_grad()
         real_images = data[0].to(device)
         b_size = real_images.size(0) 
         label = torch.ones(b_size).to(device)
         output = D(real_images).view(-1)

         real_loss = loss_fn(output, label)
         real_loss.backward()
         D_x = output.mean().item()

         #generate fakeimg
         noise = torch.randn(b_size, nz, 1, 1, device=device)
         fake_images = G(noise)
         label.fill_(0)
         output = D(fake_images.detach()).view(-1)
         fake_loss = loss_fn(output, label)
         fake_loss.backward()
         D_G_z1 = output.mean().item()
         loss_D = real_loss + fake_loss
         d_optimizer.step()

         #
         G.zero_grad()
         label.fill_(1)
         output = D(fake_images).view(-1)
         loss_G = loss_fn(output, label)
         loss_G.backward()

         D_G_z2 = output.mean().item()
         g_optimizer.step()


         if i % 50 == 0:
             print("epoch[{}/{}], step[{}/{}], loss_D:{:.4f}, loss_g:{:.4f}, D(x):{:.2f}, D(G(z)):{:.2f}"
                .format(epoch, num_epochs, i, total_step, loss_D.item(), loss_G.item(),
                D_x, D_G_z1, D_G_z2))
#%%
with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
#%%
real_batch = next(iter(dataloader))

plt.figure(figsize=(30, 30))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("real image")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True,).cpu(),(1,2,0)))

plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("real image")
plt.imshow(np.transpose(vutils.make_grid(fake, padding=2,normalize=True,(1,2,0)))
plt.show()
