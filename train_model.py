import argparse
import os
import numpy as np
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from datasets import ExampleDataset, MultiEpochsDataLoader, AverageMeter
from discriminator import weights_init_normal, SFCNDiscriminator
from generator import DiffeoGenerator
from losses import r1_reg, smooth_loss_l2


random.seed(1337)

# Tensor type
Tensor = torch.cuda.FloatTensor


def train(opt):
    os.makedirs("saved_models", exist_ok=True)

    device = torch.device("cuda:0")

    # Loss functions
    criterion_cls = F.binary_cross_entropy_with_logits
    criterion_regression = F.mse_loss

    lambda_gp = 1

    nb_features = [
        [16, 32, 32, 32, 32],
        [32, 32, 32, 32, 32, 32, 16, 16]  # decoder
    ]

    generator = DiffeoGenerator(inshape=(opt.img_height, opt.img_height, opt.img_height),
                                c_dim=2,
                                use_probs=False,
                                int_downsize=2,
                                in_channels=1,
                                int_steps=7,
                                nb_unet_features=nb_features).to(device)

    # Multi-GPU training -- Probably needed
    if opt.n_gpu > 1:
        generator = nn.DataParallel(generator, list(range(opt.n_gpu)))

    discriminator = SFCNDiscriminator(in_channels=1,
                                      channel_number=[16, 32, 64, 64, 64, 128, 64]).to(device)
    # Multi-GPU training
    if opt.n_gpu > 1:
        discriminator = nn.DataParallel(discriminator, list(range(opt.n_gpu)))
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    # Discriminator needs to be trained more slowly -- this is not critical, but gives best results
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.2 * opt.lr, betas=(opt.b1, opt.b2))

    dataset = ExampleDataset('train', aug_p=opt.aug_p)
    dataloader = MultiEpochsDataLoader(dataset,
                                       batch_size=opt.batch_size,
                                       drop_last=False,
                                       shuffle=True,
                                       num_workers=opt.n_cpu,
                                       pin_memory=True)

    for epoch in range(opt.n_epochs):

        # It looks ugly like this, but I think it helps understand what each loss does a bit better
        # Adversarial, class/regression losses and regularisation
        discriminator_adv_loss = []
        discriminator_cls_loss = []
        discriminator_reg_loss = []

        # Adversarial, class/regression losses and regularisation
        generator_adv_loss = []
        generator_cls_loss = []
        generator_reg_loss = []

        with tqdm(total=len(dataloader.dataset)) as progress_bar:
            for i, (imgs, class_labels, reg_labels) in enumerate(dataloader):

                # Model inputs
                x_real = Variable(imgs.type(Tensor))
                class_labels = Variable(class_labels.type(Tensor))
                reg_labels = Variable(reg_labels.type(Tensor))

                '''
                We sample transformation by just shuffling the true labels around
                '''
                rand_idx = torch.randperm(class_labels.size(0))
                sampled_class_labels = class_labels[rand_idx]
                sampled_reg_labels = reg_labels[rand_idx]
                # For the regression targets, we want the change from real to sampled label
                # E.g. if the original brain age is 50 and now we get 80, then what is passed to the model is 30
                diff_labels = (sampled_reg_labels - reg_labels)
                # The target labels are the classes and regression targets
                target_labels = torch.cat([sampled_class_labels, sampled_reg_labels], dim=1)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()

                x_real.requires_grad_()
                # Compute loss with real images.
                out_real, out_cls, out_reg = discriminator(x_real)
                # Non-saturating adversarial loss
                d_loss_real = torch.mean(F.softplus(-out_real))
                # How good is the model at predicting classification and regression
                d_loss_cls = criterion_cls(out_cls, class_labels) + criterion_regression(out_reg, reg_labels)
                d_loss_cls = torch.mean(d_loss_cls)

                # Create the fake images -- and see how much they fool the discriminator
                x_fake, preint_flow = generator(x_real, target_labels)
                # Use detach here because we don't want this loss to send gradients to Generator
                out_fake, _ = discriminator(x_fake.detach())
                d_loss_fake = torch.mean(F.softplus(out_fake))
                # Compute loss for gradient penalty.
                d_loss_gp = r1_reg(out_real, x_real)
                d_adv = d_loss_real + d_loss_fake

                discriminator_adv_loss.append(d_adv.item())
                discriminator_cls_loss.append(d_loss_cls.item())
                discriminator_reg_loss.append(d_loss_gp.item())

                d_loss = d_adv + d_loss_cls + lambda_gp * d_loss_gp

                d_loss.backward()
                optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------

                # Every n_critic times update generator
                if i % opt.n_critic == 0:
                    optimizer_G.zero_grad()

                    out_src, out_cls = discriminator(x_fake)
                    g_loss_fake = torch.mean(F.softplus(-out_src))
                    g_loss_cls = criterion_cls(out_cls, sampled_class_labels) + criterion_regression(out_reg, sampled_reg_labels)

                    generator_adv_loss.append(g_loss_fake.item())
                    generator_cls_loss.append(g_loss_cls.item())

                    # This is the loss used to ensure displacements are smooth
                    int_loss_l2 = smooth_loss_l2(preint_flow)

                    generator_reg_loss.append(int_loss_l2.item())

                    g_loss = g_loss_fake + g_loss_cls + 10 * int_loss_l2
                    g_loss.backward()
                    optimizer_G.step()

                # --------------
                #  Log Progresssion
                # --------------

                progress_bar.set_postfix(d_adv=np.mean(discriminator_adv_loss),
                                         d_cls=np.mean(discriminator_cls_loss),
                                         g_adv=np.mean(generator_adv_loss),
                                         g_cls=np.mean(generator_cls_loss),
                                         g_int=np.mean(generator_reg_loss))
                progress_bar.update(x_real.size(0))

        if epoch % 2 == 0:
            torch.save(generator.state_dict(), f"saved_models_/generator_{epoch}.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")

    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--aug_p", type=float, default=0.8, help="amount of augmentation to use")
    parser.add_argument("--n_cpu", type=int, default=20, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=int, default=8, help="number of gpus on machine to use")
    parser.add_argument("--img_height", type=int, default=128, help="size of image height")
    parser.add_argument("--n_critic", type=int, default=1, help="number of training iterations for discriminator")
    opt = parser.parse_args()

    train(opt)
