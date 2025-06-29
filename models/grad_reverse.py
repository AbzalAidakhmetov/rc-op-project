import torch
from torch.autograd import Function
from typing import cast

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.clone()

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lambd * grad_out, None

def grad_reverse(x: torch.Tensor, lambd: float = 1.) -> torch.Tensor:
    return cast(torch.Tensor, GradReverse.apply(x, lambd)) 