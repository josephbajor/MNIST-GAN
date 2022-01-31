import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import rand_label, rand_label_batched
from model import ConvDescriminator, ConditionalGenerator
from dataloaders import load_data
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid



def train(hparams, load:bool = False):
    savepath = hparams.model_path
    dsc = ConvDescriminator(hparams).to(hparams.device).double()
    gen = ConditionalGenerator(hparams).to(hparams.device).double()

    if load:
        assert os.path.isfile(savepath), f"Cannot find {savepath}!"
        dsc.load_state_dict(torch.load(savepath))
        gen.load_state_dict(torch.load(savepath))

    dsc_opt = torch.optim.Adam(dsc.parameters(), lr=hparams.learn_rate)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=hparams.learn_rate)
    loss_fn = nn.BCELoss()

    tboard_writer = SummaryWriter(savepath)
    
    trainloader, testloader = load_data(hparams)

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(hparams.epochs):
        total_loss = 0.0
        for label, x_true in trainloader:
            x_true = x_true.to(hparams.device)
            label = label.to(hparams.device)
            label = F.one_hot(label)
            label_gen = rand_label_batched(hparams, num_labels=10)
            label_gen = label_gen.to(hparams.device)

            dsc.zero_grad()
            gen.zero_grad()
            #z = torch.randn(hparams.batch_size, hparams.z_dim, 1, 1).to(hparams.device)
            x_gen = gen(label_gen)
            dsc_real = dsc(x_true)
            dsc_loss_real = loss_fn(dsc_real, label.double())
            dsc_fake = dsc(x_gen)
            dsc_loss_fake = loss_fn(dsc_fake,label_gen)
            dsc_loss = (dsc_loss_fake + dsc_loss_real) / 2
            #dsc_loss.backward(retain_graph=True)
            dsc_loss.backward()
            dsc_opt.step()

            gen_loss = loss_fn(dsc_fake, label_gen)
            gen_loss.backward()
            gen_opt.step()

            total_loss += gen_loss.item() + dsc_loss.item()

        print(f"\nLoss for Epoch {epoch+1}: {total_loss}")

        with torch.no_grad():
            x_gen = gen(label_gen)
            img_grid = make_grid(x_gen)
            tboard_writer.add_image("generated", img_grid, global_step = epoch)

    torch.save(dsc.state_dict(), savepath)
    torch.save(gen.state_dict(), savepath)
    print(f"Saving model parameters to {savepath}")