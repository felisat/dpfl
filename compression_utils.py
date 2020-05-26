import torch
from functools import partial

def identity(T):
	return T

def rand_k(T, p=0.1):
    k = max(1, int(torch.numel(T)*p))
    mask = (torch.randperm(torch.numel(T)).view(T.shape)<1000).float()
    return T*mask

def top_k(T, p=0.1):
    k = max(1, int(torch.numel(T)*p))
    v = torch.topk(torch.abs(T).view(-1), k)[0][-1]
    mask = (torch.abs(T)>=v).float() 
    return T*mask

def gaussian_mechanism(T, sigma=0.001):
    n = torch.norm(T)
    return T + n*sigma*torch.randn_like(T)


def get_compression(compression):
	if not compression:
		return identity
	else: 
		name, hp = compression
		return partial(globals()[name], **hp)
