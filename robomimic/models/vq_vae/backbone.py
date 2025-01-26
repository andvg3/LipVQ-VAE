import torch
import torch.nn as nn
import torch.nn.functional as F


class VQVAE(nn.Module):
    def __init__(
        self, feature_dim, latent_dim, num_embeddings=128, commitment_cost=0.25
    ):
        super(VQVAE, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

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

        # Codebook
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        # Encoder
        z_e = self.encoder(x)  # Shape: [batch_size, latent_dim]

        # Quantization
        z_q, quantization_loss = self.quantize(z_e)
        z_latent = z_q.clone().detach()

        # Decoder
        x_recon = self.decoder(z_q)  # Reconstruct input

        # Compute VQ-VAE Loss
        recon_loss = F.mse_loss(x_recon, x)  # Reconstruction loss
        loss = recon_loss + quantization_loss

        return z_latent, loss

    def quantize(self, z_e):
        # Calculate distances between encoder output and embeddings
        z_e_expanded = z_e.unsqueeze(1)  # Shape: [batch_size, 1, latent_dim]
        distances = (
            (z_e_expanded - self.embedding.weight).pow(2).sum(-1)
        )  # [batch_size, num_embeddings]

        # Get nearest embedding indices
        encoding_indices = torch.argmin(distances, dim=1)  # Shape: [batch_size]

        # Quantized latent vectors
        z_q = self.embedding(encoding_indices)  # Shape: [batch_size, latent_dim]

        # Quantization loss
        commitment_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z_e)
        embedding_loss = F.mse_loss(z_q, z_e.detach())
        quantization_loss = embedding_loss + commitment_loss

        # Straight-through gradient estimator
        z_q = z_e + (z_q - z_e).detach()

        return z_q, quantization_loss


# Example usage
if __name__ == "__main__":
    batch_size = 80
    feature_dim = 12
    latent_dim = 208
    num_embeddings = 512

    model = VQVAE(feature_dim, latent_dim, num_embeddings)
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
