import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define action tokenizers (methods) and corresponding values, excluding "Bin"
action_tokenizers = ["MLP", "FAST", "VQ-VAE", "LFQ-VAE", "LipVQ-VAE (Ours)"]
smoothness_scores = np.array([5.71, 5.67, 4.79, 2.34, 0.63])
success_rates = np.array([0.442, 0.471, 0.475, 0.489, 0.530])

# Different markers for each tokenizer
markers = ["o", "D", "^", "v", "P"]

sns.set_style("darkgrid")
plt.figure(figsize=(8, 4))

# Plot each tokenizer
for tokenizer, smooth, success, marker in zip(
    action_tokenizers, smoothness_scores, success_rates, markers
):
    plt.scatter(
        smooth, success, s=150, marker=marker, edgecolor="black", label=tokenizer
    )

# Self-defined regression line parameters:
m = (
    -0.014
)  # slope: negative slope suggests that lower smoothness score leads to higher success rate
b = 0.527  # intercept: adjust to shift the line up/down

# Define x-range for the line
x_vals = np.linspace(min(smoothness_scores) - 1, max(smoothness_scores) + 1, 100)
y_vals = m * x_vals + b

# Plot the custom regression line
plt.plot(x_vals, y_vals, color="black", linestyle="--", linewidth=2)

# Set the font size for labels equal to that of the legend title (12)
label_fontsize = 12

plt.xlabel("Smoothness Score (↓)", fontsize=16)
plt.ylabel("Success Rate", fontsize=16)

plt.legend(
    title="Action Tokenizer",
    fontsize=label_fontsize,
    title_fontsize=label_fontsize,
    loc="lower left",
)

# Save the plot to a file
plt.savefig("action_tokenizer_comparison_custom_line.pdf", bbox_inches="tight")
plt.close()
