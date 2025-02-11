import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# Load the tensor
file_path = (
    "/home/anvuong/Desktop/robocasa/expdata/robocasa/action/proposed_action_v4.pt"
)
context_actions = torch.load(file_path)

# Ensure it's a NumPy array for t-SNE
if isinstance(context_actions, torch.Tensor):
    context_actions = context_actions.detach().cpu().numpy()

# Check shape and slice for the first 10 timesteps
num_samples, latent_dim = context_actions.shape
seq_len = 10
assert num_samples >= seq_len, "Error: Not enough data for 10 timesteps."

# Take only the first 10 timesteps
first_10_timesteps = context_actions[:seq_len]

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=8, random_state=12)
first_10_2d = tsne.fit_transform(first_10_timesteps)

# Plot t-SNE result and connect points
plt.figure(figsize=(8, 6))
plt.plot(first_10_2d[:, 0], first_10_2d[:, 1], marker="o", linestyle="-", color="b")
for i, (x, y) in enumerate(first_10_2d):
    plt.text(x, y, str(i), fontsize=12, color="red")

plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of First 10 Timesteps")
save_path = "/home/anvuong/Desktop/robocasa/expdata/robocasa/action/tsne_first_10_proposal_v4.png"
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Plot saved to {save_path}")
