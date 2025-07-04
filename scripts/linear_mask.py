import torch
import torch.nn as nn

def get_linear_mask(module:nn.Module) -> torch.Tensor:
    x = module.weight.data
    output_size, input_size = x.shape
    x_norm = torch.abs(x) / torch.sum(torch.abs(x), dim=0, keepdim=True)
    neff = torch.floor(1/torch.sum((x_norm ** 2), dim=0, keepdim=True).squeeze(0))
    
    _, indices = torch.sort(x_norm, dim=0, descending=True)
    range_tensor = torch.arange(output_size, device=x.device).unsqueeze(0).expand(input_size, -1).T
    sorted_mask = range_tensor < neff
    
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(0, indices, sorted_mask)
    return mask