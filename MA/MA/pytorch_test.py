import torch

if torch.cuda.is_available():
    print("CUDA is available. PyTorch is using GPU.")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. PyTorch is using CPU.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
