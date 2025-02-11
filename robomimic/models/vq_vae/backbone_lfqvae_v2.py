import torch
import torch.nn as nn
import torch.nn.functional as F


def lipschitz_regularization(encoder_output):
    """
    Computes Lipschitz regularization term to enforce smooth latent space transitions.
    :param encoder_output: Tensor of shape [B, hidden_dim]
    :return: Regularization loss
    """
    diffs = (
        encoder_output[:, 1:] - encoder_output[:, :-1]
    )  # Compute differences across adjacent features
    norms = torch.norm(diffs, dim=-1)  # L2 norm across feature dimension
    return torch.mean(norms)  # Penalize large variations


class LFQQuantizer(nn.Module):
    def __init__(self, num_codes, code_dim):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.codebook = nn.Parameter(
            torch.randn(num_codes, code_dim)
        )  # Learnable codebook
        nn.init.kaiming_uniform_(self.codebook)  # Proper initialization

    def forward(self, z_e):
        batch_size, latent_dim = z_e.shape  # Ensure shape consistency
        z_e_expanded = z_e.unsqueeze(1)  # Shape: [B, 1, D]
        codebook_expanded = self.codebook.unsqueeze(0)  # Shape: [1, num_codes, D]
        distances = torch.norm(
            z_e_expanded - codebook_expanded, dim=-1
        )  # Compute L2 distances
        indices = torch.argmin(distances, dim=-1)  # Get closest code
        z_q = self.codebook[indices]  # Retrieve quantized values
        return z_q, indices


class LLFQVAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, num_codes=128, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.quantizer = LFQQuantizer(num_codes, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
        )
        self.to_output = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        h = self.encoder(x)  # Correct shape: [B, hidden_dim]
        z_e = self.to_latent(h)  # Shape: [B, latent_dim]
        z_q, indices = self.quantizer(z_e)  # Shape: [B, latent_dim]
        z_latent = z_q.clone().detach()
        recon = self.decoder(z_q)  # Correct shape: [B, hidden_dim]
        x_recon = self.to_output(recon)  # Shape: [B, feature_dim]

        # Compute losses
        recon_loss = F.mse_loss(x_recon, x)  # Reconstruction loss
        commitment_loss = F.mse_loss(z_q.detach(), z_e)  # Commitment loss
        codebook_loss = F.mse_loss(z_q, z_e.detach())  # Codebook loss
        lipschitz_loss = lipschitz_regularization(h)  # Lipschitz loss

        loss = recon_loss + 0.25 * commitment_loss + 0.25 * codebook_loss
        return z_latent, loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 80
    feature_dim = 12
    latent_dim = 208
    num_codes = 128
    model = LLFQVAE(feature_dim, latent_dim, num_codes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = torch.randn(batch_size, feature_dim).to(device)

    for epoch in range(1000):
        optimizer.zero_grad()
        z_latent, loss = model(data)
        print(f"Epoch {epoch}: Latent Shape {z_latent.shape}, Loss {loss.item():.4f}")
        loss.backward()
        optimizer.step()
