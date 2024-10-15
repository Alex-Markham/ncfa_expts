!git clone https://github.com/xwshen51/DEAR.git
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from bgm import BGM
from sagan import Discriminator

def run_bgm_discovery(dataset_path, latent_dim=10, num_epochs=100, lr=0.001, batch_size=32):
    # use gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    dataset = np.loadtxt(dataset_path, delimiter=",")
    dataset = torch.FloatTensor(dataset)

    # Create DataLoader
    dataloader = DataLoader(TensorDataset(dataset), batch_size=batch_size, shuffle=True)

    # Initialize the model bgm
    model = BGM(latent_dim, 64, dataset.shape[1], 'gaussian', 'resnet', 256, 10, 'gaussian', 'gaussian', dataset.shape[1], None).to(device)
    discriminator = Discriminator(64, dataset.shape[1]).to(device)

    # Set up optimizers
    encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=lr)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    # Training
    for epoch in range(num_epochs):
        total_loss_d = 0
        total_loss_encoder = 0
        total_loss_decoder = 0
        for batch in dataloader:
            batch = batch[0].to(device)

            # Train discriminator
            discriminator.zero_grad()
            z = torch.randn(batch.size(0), latent_dim, device=device)
            z_fake, x_fake, _ = model(batch, z)

            encoder_score = discriminator(batch, z_fake.detach())
            decoder_score = discriminator(x_fake.detach(), z)

            loss_d = F.softplus(decoder_score).mean() + F.softplus(-encoder_score).mean()
            loss_d.backward()
            D_optimizer.step()

            # Train encoder and decoder
            model.zero_grad()
            z_fake, x_fake, z_fake_mean = model(batch, z)

            encoder_score = discriminator(batch, z_fake)
            loss_encoder = encoder_score.mean()
            loss_encoder.backward()
            encoder_optimizer.step()

            model.zero_grad()
            decoder_score = discriminator(x_fake, z)
            loss_decoder = -decoder_score.mean()
            loss_decoder.backward()
            decoder_optimizer.step()

            total_loss_d += loss_d.item()
            total_loss_encoder += loss_encoder.item()
            total_loss_decoder += loss_decoder.item()

        avg_loss_d = total_loss_d / len(dataloader)
        avg_loss_encoder = total_loss_encoder / len(dataloader)
        avg_loss_decoder = total_loss_decoder / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Discriminator Loss: {avg_loss_d:.4f}, Encoder Loss: {avg_loss_encoder:.4f}, Decoder Loss: {avg_loss_decoder:.4f}')

    # Extract biadj from the trained model
    biadj = model.prior.A.detach().cpu().numpy()

    return biadj

biadj = run_bgm_discovery(dataset_path, latent_dim=10, num_epochs=100, lr=0.001, batch_size=32)
