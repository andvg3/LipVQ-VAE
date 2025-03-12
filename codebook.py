import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define codebook sizes and corresponding success rates
codebook_sizes = [r"$2^8$", r"$2^9$", r"$2^{10}$", r"$2^{11}$"]
success_rates = np.array([0.475, 0.509, 0.530, 0.511])

# Numerical values for codebook sizes (for plotting)
codebook_x = [8, 9, 10, 11]

sns.set_style("darkgrid")
plt.figure(figsize=(8, 2))

# Plot each codebook size with "+" markers
for size, success in zip(codebook_x, success_rates):
    plt.scatter(size, success, s=150, marker="P", color="purple")

# Connect dots with a solid black line
plt.plot(codebook_x, success_rates, color="black", linestyle="-", linewidth=2)

# Set labels and ticks
plt.xlabel("Codebook Size", fontsize=14)
plt.ylabel("Success Rate", fontsize=14)
plt.xticks(codebook_x, codebook_sizes, fontsize=13)

# Save and show the plot
plt.savefig("impact_of_codebook_size_uniform.pdf", bbox_inches="tight")
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# # Define codebook sizes and corresponding success rates
# codebook_sizes = [r"$2^8$", r"$2^9$", r"$2^{10}$"]
# success_rates = np.array([0.475, 0.509, 0.530])

# # Numerical values for codebook sizes (for plotting)
# codebook_x = [8, 9, 10]

# sns.set_style("darkgrid")
# plt.figure(figsize=(8, 2))

# # Plot each codebook size with "+" markers
# for size, success in zip(codebook_x, success_rates):
#     plt.scatter(size, success, s=150, marker="P", color="purple")

# # Connect dots with a solid black line
# plt.plot(codebook_x, success_rates, color="black", linestyle="-", linewidth=2)

# # Set labels and ticks
# plt.xlabel("Codebook Size", fontsize=14)
# plt.ylabel("Success Rate", fontsize=14)
# plt.xticks(codebook_x, codebook_sizes, fontsize=13)

# # Save and show the plot
# plt.savefig("impact_of_codebook_size_uniform.pdf", bbox_inches="tight")
# plt.show()
