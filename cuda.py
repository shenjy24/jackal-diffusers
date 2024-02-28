import torch

if torch.cuda.is_available():
    print("cuda is available")
else:
    print("cuda is not available")
