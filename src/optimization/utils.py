from torch.optim import Adam, Adamax, SGD
from src.optimization.adamw import AdamW


def setup_e2e_optimizer(model, opts):
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(model.parameters(), lr=opts.learning_rate, betas=opts.betas)

    return optimizer
