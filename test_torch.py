import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a simple tensor
x = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"Tensor: {x}")

