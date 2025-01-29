import numpy as np
import torch
import torch.nn as nn

# They have affine transformation over EVERY SINGLE element, this will lead to many parameters...
# How to use: specify the dimensions that you want to layernorm over
# Ex, with 'batch patches height width channels' to layernorm over the last three pass 'height width channels'
class ConditionalLN(nn.Module):
    # Dim can be numpy array, torch.Size, or python list, of sizes
    def __init__(self, dim, eps=1e-5):
        super(ConditionalLN, self).__init__()
        self.gamma = 1
        self.beta = 0
        self.eps = eps
        if isinstance(dim, int):
            self.num_dims = 1
        else:
            self.num_dims = len(dim)
        self.dim_sizes = dim
    # We pop a 128 x 11 delta gamma and beta off queue
    def forward(self, input, delta_gamma=0, delta_beta=0):
        new_gamma = self.gamma + delta_gamma
        new_beta = self.beta + delta_beta
        mean_dims = np.arange(len(input.shape) - self.num_dims, len(input.shape))
        mean = torch.mean(input, dim=tuple(mean_dims), keepdims=True)
        variance = torch.var(input, dim=tuple(mean_dims), keepdims=True, unbiased=False)
        normalized_input = (input - mean) / torch.sqrt(variance + self.eps)
        if (isinstance(delta_gamma, torch.Tensor)):
            new_gamma = torch.unsqueeze(torch.unsqueeze(new_gamma, 1), 2)
            new_beta = torch.unsqueeze(torch.unsqueeze(new_beta, 1), 2)
       
        return torch.mul(normalized_input, new_gamma) + new_beta