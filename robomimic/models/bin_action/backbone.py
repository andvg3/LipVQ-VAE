import torch
import torch.nn as nn


class AdaptiveBinActionEmbedding(nn.Module):
    def __init__(
        self, action_dim, output_dim, num_bins=20, embedding_dim=64, num_step_stop=10000
    ):
        super(AdaptiveBinActionEmbedding, self).__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim

        # Initialize running min and max for each action dimension
        self.register_buffer("running_min", torch.full((action_dim,), float("inf")))
        self.register_buffer("running_max", torch.full((action_dim,), float("-inf")))

        # Create an embedding layer for each action dimension
        self.embedding_layers = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=num_bins, embedding_dim=embedding_dim)
                for _ in range(action_dim)
            ]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim * action_dim, embedding_dim * action_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim * action_dim // 2, output_dim),
            nn.GELU(),
        )

        # Create flag for stop updating after a certain number of steps
        self._num_step = 0
        self._num_step_stop = num_step_stop
        self._update_enabled = True

    def update_running_stats(self, actions):
        # Update running min and max for each action dimension
        self.running_min = torch.minimum(self.running_min, actions.min(dim=0)[0])
        self.running_max = torch.maximum(self.running_max, actions.max(dim=0)[0])

    def compute_bins(self):
        # Compute bin boundaries for each action dimension
        bin_boundaries = []
        for i in range(self.action_dim):
            min_val = self.running_min[i]
            max_val = self.running_max[i]
            boundaries = torch.linspace(
                min_val, max_val, self.num_bins + 1, device=self.running_min.device
            )
            bin_boundaries.append(boundaries)
        return bin_boundaries

    def discretize(self, actions):
        # Compute bin boundaries
        bin_boundaries = self.compute_bins()

        # Discretize the continuous actions into bin indices
        bin_indices = []
        for i in range(self.action_dim):
            boundaries = bin_boundaries[i]
            # Find the bin index for each action in the batch
            indices = torch.bucketize(actions[:, i], boundaries)
            # Clamp indices to valid range [0, num_bins - 1]
            indices = torch.clamp(indices - 1, 0, self.num_bins - 1)
            bin_indices.append(indices)
        return torch.stack(bin_indices, dim=1)

    def forward(self, actions):
        # Update running statistics
        if self._update_enabled:
            self.update_running_stats(actions)
            self._num_step += 1
            if self._num_step >= self._num_step_stop:
                self._update_enabled = False

        # Discretize the actions
        bin_indices = self.discretize(actions)

        # Embed each action dimension
        embeddings = []
        for i in range(self.action_dim):
            embeddings.append(self.embedding_layers[i](bin_indices[:, i]))

        # Concatenate the embeddings along the last dimension
        action_embeddings = torch.cat(embeddings, dim=-1)
        action_embeddings = self.output_layer(action_embeddings)

        return action_embeddings


if __name__ == "__main__":
    batch_size = 4
    action_dim = 2
    num_bins = 5
    embedding_dim = 8

    # Specify the path where you want to save the model
    save_path = "adaptive_bin_action_embedding.pth"

    embedding_layer = AdaptiveBinActionEmbedding(action_dim, num_bins, embedding_dim)
    embedding_layer.load_state_dict(torch.load(save_path))
    print(embedding_layer.running_max)
    for i in range(100):
        # Create a random batch of actions
        actions = torch.rand(4, 2)

        # Initialize the adaptive embedding layer

        # Get the embeddings
        embeddings = embedding_layer(actions)

    # Save the model's state dictionary
