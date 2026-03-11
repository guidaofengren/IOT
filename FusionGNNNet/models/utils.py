"""
models/utils.py — Model utility functions (init, helpers).
Kept from NexusNet baseline.
"""
import torch
from torch.nn import init
import numpy as np


def normalize_adj(adj, mode='sym'):
    inv_sqrt_degree = 1. / (torch.sqrt(adj.abs().sum(dim=-1, keepdim=False)) + 1e-10)
    if len(adj.shape) == 3:
        return inv_sqrt_degree[:, :, None] * adj * inv_sqrt_degree[:, None, :]
    return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]


def glorot_weight_zero_bias(model):
    for module in model.modules():
        if hasattr(module, "weight"):
            if not ("BatchNorm" in module.__class__.__name__):
                init.xavier_uniform_(module.weight, gain=1)
            else:
                init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                init.constant_(module.bias, 0)


class Expression(torch.nn.Module):
    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
            self.__class__.__name__
            + "("
            + "expression="
            + str(expression_str)
            + ")"
        )


def np_to_var(X, requires_grad=False, dtype=None, pin_memory=False, **tensor_kwargs):
    if not hasattr(X, "__len__"):
        X = [X]
    X = np.asarray(X)
    if dtype is not None:
        X = X.astype(dtype)
    X_tensor = torch.tensor(X, requires_grad=requires_grad, **tensor_kwargs)
    if pin_memory:
        X_tensor = X_tensor.pin_memory()
    return X_tensor

class Conv2dWithConstraint(torch.nn.Conv2d):
    def __init__(self, *args, max_norm=1., **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)
