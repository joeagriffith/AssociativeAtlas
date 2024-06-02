import torch
import math

def cosine_schedule(base, end, T):
    return end - (end - base) * ((torch.arange(0, T, 1) * math.pi / T).cos() + 1) / 2

def get_optimiser(model, optimiser, lr, wd, exclude_bias=True, exclude_bn=True, momentum=0.9, betas=(0.9, 0.999)):
    non_decay_parameters = []
    decay_parameters = []   
    for n, p in model.named_parameters():
        if exclude_bias and 'bias' in n:
            non_decay_parameters.append(p)
        elif exclude_bn and 'bn' in n:
            non_decay_parameters.append(p)
        else:
            decay_parameters.append(p)
    non_decay_parameters = [{'params': non_decay_parameters, 'weight_decay': 0.0}]
    decay_parameters = [{'params': decay_parameters}]

    assert optimiser in ['AdamW', 'SGD'], 'optimiser must be one of ["AdamW", "SGD"]'
    if optimiser == 'AdamW':
        if momentum != 0.9:
            print('Warning: AdamW does not accept momentum parameter. Ignoring it. Please specify betas instead.')
        optimiser = torch.optim.AdamW(decay_parameters + non_decay_parameters, lr=lr, weight_decay=wd, betas=betas)
    elif optimiser == 'SGD':
        if betas != (0.9, 0.999):
            print('Warning: SGD does not accept betas parameter. Ignoring it. Please specify momentum instead.')
        optimiser = torch.optim.SGD(decay_parameters + non_decay_parameters, lr=lr, weight_decay=wd, momentum=momentum)
    
    return optimiser
