from torch.autograd import Function
import torch
import halideslice


class SliceFunctionBasedOnGradientHalide(Function):

    @staticmethod
    def forward(ctx, coeff, guide):
        ctx.save_for_backward(coeff, guide)
        return halideslice.forward(coeff, guide)

    @staticmethod
    def backward(ctx, grad_output):
        coeff, guide = ctx.saved_tensors
        grad_coeff, grad_guide = halideslice.backward(
            grad_output, coeff, guide)
        return grad_coeff, grad_guide
