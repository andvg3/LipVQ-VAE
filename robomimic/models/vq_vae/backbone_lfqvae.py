import torch
import torch.nn as nn
import torch.nn.functional as F


class LFQVAE(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super(LFQVAE, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # Encoder and Decoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # Encoder
        z_e = self.encoder(x)  # Shape: [batch_size, latent_dim]

        # LFQ Quantization
        z_q = self.lfq_quantize(z_e)
        z_latent = z_q.clone().detach()

        # Decoder
        x_recon = self.decoder(z_q)  # Reconstruct input

        # Compute Loss
        recon_loss = F.mse_loss(x_recon, x)  # Reconstruction loss
        loss = recon_loss

        return z_latent, loss

    def lfq_quantize(self, z_e):
        # Normalize input to unit sphere
        z_q = F.normalize(z_e, p=2, dim=-1)
        return z_q


# Example usage
if __name__ == "__main__":
    batch_size = 80
    feature_dim = 12
    latent_dim = 208

    model = LFQVAE(feature_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Dummy data
    data = torch.randn(batch_size, feature_dim)

    # Training loop
    for epoch in range(10):
        optimizer.zero_grad()
        z_latent, loss = model(data)
        print(z_latent.data.shape)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
