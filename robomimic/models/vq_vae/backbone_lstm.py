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

        # LSTM-based Encoder
        self.encoder_lstm = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.GELU(),
            nn.LSTM(
                input_size=latent_dim,
                hidden_size=latent_dim,
                num_layers=3,
                batch_first=True,
            ),
        )

        # LSTM-based Decoder
        self.decoder_lstm = nn.Sequential(
            nn.Linear(latent_dim, feature_dim),
            nn.GELU(),
            nn.LSTM(
                input_size=feature_dim,
                hidden_size=feature_dim,
                num_layers=3,
                batch_first=True,
            ),
        )

        # Codebook for quantization
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        # Reshape input to [batch_size, seq_len, feature_dim] -> [8, 10, 12]
        x_reshaped = x.view(8, 10, self.feature_dim)

        # Encoder
        z_e, _ = self.encoder_lstm(x_reshaped)  # Shape: [8, 10, latent_dim]

        # Quantization
        z_q, quantization_loss = self.quantize(z_e)
        z_latent = z_q.clone().detach()
        z_latent = z_latent.reshape(80, -1)

        # Decoder
        x_recon, _ = self.decoder_lstm(
            z_q
        )  # Reconstruct input (Shape: [8, 10, feature_dim])

        # Reshape output back to original shape [80, 12]
        x_recon_reshaped = x_recon.reshape(80, self.feature_dim)

        # Compute VQ-VAE Loss
        recon_loss = F.mse_loss(x_recon_reshaped, x)  # Reconstruction loss
        loss = recon_loss + quantization_loss

        return z_latent, loss

    def quantize(self, z_e):
        # Calculate distances between encoder output and embeddings
        z_e_expanded = z_e.unsqueeze(2)  # Shape: [batch_size, seq_len, 1, latent_dim]
        distances = (
            (z_e_expanded - self.embedding.weight).pow(2).sum(-1)
        )  # Shape: [batch_size, seq_len, num_embeddings]

        # Soft quantization (using softmax for better gradient flow)
        q = F.softmax(-distances, dim=2)  # Shape: [batch_size, seq_len, num_embeddings]
        z_q = torch.matmul(
            q, self.embedding.weight
        )  # Shape: [batch_size, seq_len, latent_dim]

        # Quantization loss (commitment cost and embedding loss)
        commitment_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z_e)
        embedding_loss = F.mse_loss(z_q, z_e.detach())
        quantization_loss = embedding_loss + commitment_loss

        # Straight-through gradient estimator
        z_q = z_e + (z_q - z_e).detach()

        return z_q, quantization_loss
