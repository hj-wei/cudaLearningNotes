import torch



A = torch.randn((10000, 10000),dtype=torch.float32, device='cuda')
B = torch.randn((10000, 10000),dtype=torch.float32, device='cuda')
res = A @ B
print(res)