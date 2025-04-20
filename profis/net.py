import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Profis(nn.Module):
    def __init__(
        self,
        fp_size=2048,
        fc1_size=1024,
        fc2_size=1024,
        latent_size=32,
        hidden_size=512,
        gru_layers=3,
        eps_coef=1,
        dropout=0,
        alphabet_size=30,
    ):
        super(Profis, self).__init__()

        self.fp_size = fp_size
        self.fc1 = nn.Linear(fp_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc31 = nn.Linear(fc2_size, latent_size)
        self.fc32 = nn.Linear(fc2_size, latent_size)
        self.fc4 = nn.Linear(latent_size, 256)
        self.gru = nn.GRU(
            256, hidden_size, gru_layers, batch_first=True, dropout=dropout
        )
        self.fc5 = nn.Linear(hidden_size, alphabet_size)

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.softmax = nn.Softmax(dim=1)
        self.eps_coef = eps_coef

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        mu = self.fc31(h2)
        logvar = self.fc32(h2)
        return mu, logvar

    def sampling(self, z_mean, z_logvar):
        epsilon = self.eps_coef * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):
        z = self.selu(self.fc4(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 100, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = self.softmax(self.fc5(out_reshape))
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar


class VaeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_hat, y, z_mean, z_logvar):
        xent_loss = F.binary_cross_entropy(x_hat, y, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        return xent_loss, kl_loss
