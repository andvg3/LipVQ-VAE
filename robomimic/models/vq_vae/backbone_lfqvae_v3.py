import torch
import torch.nn.functional as F


def normalization(Wi, softplus_ci):  # L-inf norm
    absrowsum = torch.sum(torch.abs(Wi), dim=1, keepdim=True)  # Shape: (out_dim, 1)
    scale = torch.minimum(
        torch.tensor(1.0, device=Wi.device),
        F.softplus(softplus_ci).unsqueeze(1) / absrowsum,
    )
    return Wi * scale  # Broadcasting should now work


class LipschitzMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(out_dim, in_dim))
        self.b = torch.nn.Parameter(torch.zeros(out_dim))
        self.ci = torch.nn.Parameter(torch.ones(out_dim))  # Learnable ci parameter

    def forward(self, x):
        W_norm = normalization(self.W, self.ci)
        return torch.sigmoid(torch.matmul(x, W_norm.T) + self.b)


class LFQVAE_V3(torch.nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super(LFQVAE_V3, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # Encoder and Decoder
        self.encoder = torch.nn.Sequential(
            LipschitzMLP(feature_dim, 64),
            torch.nn.ReLU(),
            LipschitzMLP(64, 128),
            torch.nn.ReLU(),
            LipschitzMLP(128, latent_dim),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            LipschitzMLP(latent_dim, 128),
            torch.nn.ReLU(),
            LipschitzMLP(128, 64),
            torch.nn.ReLU(),
            LipschitzMLP(64, feature_dim),
            torch.nn.ReLU(),
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
    for epoch in range(10000):
        optimizer.zero_grad()
        z_latent, loss = model(data)
        # print(z_latent.data.shape)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
