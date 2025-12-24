import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim = 160, latent_dim = 2):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Linear(64, latent_dim)
        )

        self.decoder = nn. Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim)
        )

        def forward(self, x):
            z = self.encoder(x)
            reconstruction = self.decoder(z)
            return reconstruction
