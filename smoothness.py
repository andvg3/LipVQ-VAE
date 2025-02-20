import torch
import numpy as np
import os


def least_energy_smoothness(X):
    """
    Compute the Least Energy metric from the paper "Least Energy Paths for Motion Planning".

    Args:
        X (np.ndarray): Input sequence of shape (seq_len, hidden_dim).

    Returns:
        float: Least Energy measure (lower is smoother).
    """
    seq_len, _ = X.shape
    if seq_len < 3:
        return 0.0  # Not enough points for second derivatives

    energy = np.sum(np.linalg.norm(X[2:] - 2 * X[1:-1] + X[:-2], axis=1) ** 2)
    return energy


# List of action files
action_files = [
    "bin_action.pt",
    "fast_action_v2.pt",
    "lfq_vae_action.pt",
    "mlp_action.pt",
    "proposed_action_v3.pt",
]

# Base directory
base_dir = "expdata/robocasa/action"

# Compute Least Energy for each action file
for filename in action_files:
    tensor_path = os.path.join(base_dir, filename)

    try:
        # Load PyTorch tensor
        tensor = torch.load(tensor_path).detach().cpu()

        # Convert to NumPy array
        X = tensor.numpy()

        # Compute Least Energy smoothness
        energy_value = least_energy_smoothness(X)
        print(f"Least Energy Smoothness for {filename}: {energy_value}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
