from torch.autograd import Function
import torch
import time
import slice
import sys
import halideslice


class SliceFunction(Function):

    @staticmethod
    def forward(ctx, coeff, guide):

        _, ch, cw, cd, cc = coeff.size()
        _, gh, gw = guide.size()

        i = torch.arange(ch, dtype=torch.float32, device=coeff.device)
        j = torch.arange(cw, dtype=torch.float32, device=coeff.device)
        k = torch.arange(cd, dtype=torch.float32, device=coeff.device)

        x = torch.arange(gh, dtype=torch.float32, device=coeff.device)
        y = torch.arange(gw, dtype=torch.float32, device=coeff.device)

        sh = ch / gh
        sw = cw / gw

        # [gh, ch]
        t1 = SliceFunction.tau(
            x.view(1, -1) * sh - i.view(-1, 1))
        # [gw, cw]
        t2 = SliceFunction.tau(
            y.view(1, -1) * sw - j.view(-1, 1))
        # [_, gh, gw, cd]
        t3 = SliceFunction.tau(
            guide.view(-1, gh, gw, 1) * 8 - k.view(1, 1, 1, -1))

        t1 = t1.view(-1, gh, 1, ch, 1, 1, 1)
        t2 = t2.view(-1, 1, gw, 1, cw, 1, 1)
        t3 = t3.view(-1, gh, gw, 1, 1, cd, 1)

        ctx.save_for_backward(coeff, guide, t1, t2, t3)

        # [_, gh, gw, ch, cw, cd, cc]
        t4 = t1 * t2 * t3 * coeff.view(-1, 1, 1, ch, cw, cd, cc)
        return torch.sum(t4, dim=[3, 4, 5])

    @staticmethod
    def backward(ctx, grad_output):
        coeff, guide, t1, t2, t3 = ctx.saved_tensors

        # dtua_t3 = torch.where(t3 >= 0 and t3 <= 1, torch.full_like(t3, -1), 0)
        # + torch.where(t3 <= 0 and t3 >= -1, torch.full_like(t3, 1), 0)

        dtua_t3 = torch.where(
            torch.abs(t3) < 1, -torch.sign(t3), torch.full_like(t3, 0))

        _, ch, cw, cd, cc = coeff.size()
        coeff = coeff.view(-1, 1, 1, ch, cw, cd, cc)

        _, gh, gw, gc = grad_output.size()
        grad_output = grad_output.view(-1, gh, gw, 1, 1, 1, gc)

        # [_, gh, gw, ch, cw, cd, cc]
        t4 = grad_output * t1 * t2 * dtua_t3 * 8 * coeff
        t5 = grad_output * t1 * t2 * t3

        dgxy = torch.sum(t4, dim=[3, 4, 5, 6])
        daijkc = torch.sum(t5, dim=[1, 2])

        return daijkc, dgxy

    @staticmethod
    def tau(x):
        x = 1 - torch.abs(x)
        return torch.where(x > 0, x, torch.full_like(x, 0))


class SliceFunctionBasedCuda(Function):

    @staticmethod
    def forward(ctx, coeff, guide):
        ctx.save_for_backward(coeff, guide)
        return slice.forward(coeff, guide)

    @staticmethod
    def backward(ctx, grad_output):
        coeff, guide = ctx.saved_tensors
        grad_coeff, grad_guide = slice.backward(grad_output, coeff, guide)
        return grad_coeff, grad_guide


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


def main(argv):

    batch, height, width = int(argv[1]), int(argv[2]), int(argv[3])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coeff = torch.randn(batch, 16, 16, 8, 12,
                        device=device, requires_grad=True)
    guide = torch.randn(batch, height, width,
                        device=device, requires_grad=True)
    grad_output = torch.randn(batch, height, width, 12, device=device)

    SliceFunctionApply = SliceFunction.apply
    with torch.autograd.profiler.profile(use_cuda=True) as profiler:
        for _ in range(10):
            output = SliceFunctionApply(coeff, guide)
            output.backward(grad_output)
    print(profiler.key_averages())

    SliceFunctionBasedCudaApply = SliceFunctionBasedCuda.apply
    with torch.autograd.profiler.profile(use_cuda=True) as profiler:
        for _ in range(10):
            output = SliceFunctionBasedCudaApply(coeff, guide)
            output.backward(grad_output)
    print(profiler.key_averages())

    SliceFunctionBasedOnGradientHalideApply = SliceFunctionBasedOnGradientHalide.apply
    with torch.autograd.profiler.profile(use_cuda=True) as profiler:
        for _ in range(10):
            output = SliceFunctionBasedOnGradientHalideApply(coeff, guide)
            output.backward(grad_output)
    print(profiler.key_averages())


if __name__ == "__main__":
    main(sys.argv)
