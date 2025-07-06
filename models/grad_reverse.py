from torch.autograd import Function
import torch.nn as nn
import torch
from typing import cast

class GradReverse(Function):
    """
    Gradient Reversal Layer from DANN (Domain-Adversarial Neural Network)
    """
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient and scale by lambda
        return (grad_output.neg() * ctx.lambd), None

class GradientReversal(nn.Module):
    """
    A module wrapper for the Gradient Reversal Layer.
    During the forward pass, it's an identity function.
    During the backward pass, it reverses and scales the gradient.
    """
    def __init__(self):
        super(GradientReversal, self).__init__()
    
    def forward(self, x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        return cast(torch.Tensor, GradReverse.apply(x, lambd)) 