import torch
from torch.autograd import Function


class SliceFunction(Function):

    @staticmethod
    def forward(ctx, coeff, guide):

        ctx.save_for_backward(coeff)
        ctx.save_for_backward(guide)

        _, ch, cw, cd, cc = coeff.size()
        _, gh, gw = guide.size()

        i = torch.arange(ch, dtype=torch.float32)
        j = torch.arange(cw, dtype=torch.float32)
        k = torch.arange(cd, dtype=torch.float32)

        x = torch.arange(gh, dtype=torch.float32)
        y = torch.arange(gw, dtype=torch.float32)

        sh = ch / gh
        sw = cw / gw

        # [gh, ch]
        t1 = tau(x.view(1, -1) * sh - i.view(-1, 1))
        # [gw, cw]
        t2 = tau(y.view(1, -1) * sw - j.view(-1, 1))
        # [_, gh, gw, cd]
        t3 = tau(guide.view(-1, gh, gw, 1) * 8 - k.view(1, 1, 1, -1))

        t1 = t1.view(-1, gh, 1, ch, 1, 1, 1)
        t2 = t2.view(-1, 1, gw, 1, cw, 1, 1)
        t3 = t3.view(-1, gh, gw, 1, 1, cd, 1)

        # [_, gh, gw, ch, cw, cd, cc]
        t4 = t1 * t2 * t3 * coeff.view(-1, 1, 1, ch, cw, cd, cc)
        return torch.sum(t4, dim=[3, 4, 5])

    @staticmethod
    def backward(ctx, grad_output):
        coeff, guide = ctx.saved_tensors()

        _, ch, cw, cd, cc = ctx.coeff.size()
        _, gh, gw = ctx.guide.size()

        i = torch.arange(ch, dtype=torch.float32)
        j = torch.arange(cw, dtype=torch.float32)
        k = torch.arange(cd, dtype=torch.float32)

        x = torch.arange(gh, dtype=torch.float32)
        y = torch.arange(gw, dtype=torch.float32)

        sh = ch / gh
        sw = cw / gw

        # [gh, ch]
        t1 = tau(x.view(1, -1) * sh - i.view(-1, 1))
        # [gw, cw]
        t2 = tau(y.view(1, -1) * sw - j.view(-1, 1))
        # [_, gh, gw, cd]
        t3 = tau(guide.view(-1, gh, gw, 1) * 8 - k.view(1, 1, 1, -1))

        dtua_t3 = torch.where(t3 >= 0 and t3 <= 1, torch.full_like(t3, -1), 0)
        + torch.where(t3 <= 0 and t3 >= -1, torch.full_like(t3, 1), 0)

        t1 = t1.view(-1, gh, 1, ch, 1, 1, 1)
        t2 = t2.view(-1, 1, gw, 1, cw, 1, 1)
        t3 = t3.view(-1, gh, gw, 1, 1, cd, 1)
        dtua_t3 = dtua_t3.view(-1, gh, gw, 1, 1, cd, 1)

        # [_, gh, gw, ch, cw, cd, cc]
        t4 = t1 * t2 * dtua_t3 * 8 * coeff.view(-1, 1, 1, ch, cw, cd, cc)
        t5 = t1 * t2 * t3

        # dg[x,y] = [_, gh, gw, cc] * [_, gh, gw, cc]
        dgxy = grad_output[0] * torch.sum(t4, dim=[3, 4, 5])

        # dA[i,j,k,c] = [_, gh, gw, cc] * [_, gh, gw, 1]
        daxyc = grad_output[1] * torch.sum(t5, dim=[3, 4, 5])

        return [dgxy, daxyc]

    @staticmethod
    def tau(x):
        x = 1 - torch.abs(x)
        return torch.where(x > 0, x, torch.full_like(x))
