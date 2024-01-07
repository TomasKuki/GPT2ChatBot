import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU disponível. Usando GPU.")
else:
    device = torch.device("cpu")
    print("GPU não disponível. Usando CPU.")
