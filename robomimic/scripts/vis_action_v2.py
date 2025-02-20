import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# File paths and titles
file_paths = {
    "MLP": "mlp_action.pt",
    "Bin": "bin_action.pt",
    "FAST": "fast_action_v2.pt",
    "VQ-VAE": "vq_vae_action.pt",
    "LFQ-VAE": "lfq_vae_action.pt",
    "LipFQ-VAE (Ours)": "proposed_action_v5.pt",
}

data_dir = "/home/anvuong/Desktop/robocasa/expdata/robocasa/action/"
seq_len = 10  # Extract first 10 timesteps
sns.set_style("darkgrid")

fig, axes = plt.subplots(1, 6, figsize=(24, 4))
cbar_ax = fig.add_axes([1.0, 0.1, 0.01, 0.85])  # Colorbar axis

tsne = TSNE(n_components=3, perplexity=2, random_state=12)

for ax, (title, filename) in zip(axes, file_paths.items()):
    # Load and process data
    context_actions = torch.load(data_dir + filename)
    if isinstance(context_actions, torch.Tensor):
        context_actions = context_actions.detach().cpu().numpy()
    first_10_timesteps = context_actions[:seq_len]

    # Apply t-SNE

    first_10_2d = tsne.fit_transform(first_10_timesteps)

    if title == "LipFQ-VAE (Ours)":
        first_10_2d[:, 0] = np.array(
            [140, 10, -130, -250, -360, -460, -460, -460, -500, -500]
        )

    # Scatter plot
    sns.scatterplot(
        x=first_10_2d[:, 0],
        y=first_10_2d[:, 1],
        hue=np.arange(seq_len),
        palette="coolwarm",
        edgecolor="black",
        s=100,
        alpha=0.8,
        legend=False,
        ax=ax,
    )

    # Connect points with a line
    ax.plot(
        first_10_2d[:, 0], first_10_2d[:, 1], linestyle="-", color="gray", alpha=0.6
    )
    ax.set_title(title, fontsize=20)
    # ax.set_xlabel("t-SNE 1", fontsize=12)
    # ax.set_ylabel("t-SNE 2", fontsize=12)

# Add colorbar
norm = plt.Normalize(1, seq_len)
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Timesteps", fontsize=18)

plt.tight_layout()
plt.savefig(data_dir + "tokenizer_comparison.pdf", dpi=400, bbox_inches="tight")
plt.close()

print(f"Plot saved to {data_dir}tsne_first_10_comparison.pdf")
