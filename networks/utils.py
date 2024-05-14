import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ttf(x):
    """Transfer float32 tensor to device."""
    if torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x, device=device, dtype=torch.float32)
def tti(x):
    """Transfer int64 tensor to device."""
    if torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x, device=device, dtype=torch.int64)