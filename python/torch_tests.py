import torch
import time

def timer(func, input):
    torch.cuda.synchronize()
    start = time.time()
    func(input)
    torch.cuda.synchronize()
    end = time.time()
    return end - start


def run_svd(A):
    n = min(A.shape)
    U,S,Vh = torch.linalg.svd(A, full_matrices=False)
    S[0:(n // 2)] = 0.0
    Vh *= S.unsqueeze(1)
    return U @ Vh

A = torch.rand(40,16*16*16*5, dtype=torch.complex64)

for i in range(10):
    print(timer(run_svd, A))


for i in range(10):
    print(timer(run_svd, A.transpose(0,1)))
