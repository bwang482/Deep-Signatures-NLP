"""Defines the necessary operations for taking the signature of a path. See __init__.py and pytorch_siglayer.py for
further explanation.
"""

import iisignature
import torch
import torch.autograd as autograd


class path_sig_fn(autograd.Function):
    """An autograd.Function corresponding to the signature map. See also pytorch_siglayer.py's path_sig.__doc__,"""

    @staticmethod
    def forward(ctx, path, depth):
        device = path.device
        # transpose because the PyTorch convention for convolutions is channels first. The iisignature expectation is
        # that channels are last.
        path = path.detach().cpu().numpy().transpose()
        ctx.path = path
        ctx.depth = depth
        return torch.tensor(iisignature.sig(path, depth), dtype=torch.float, device=device)

    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        backprop = iisignature.sigbackprop(grad_output.cpu().numpy(), ctx.path, ctx.depth)
        # transpose again to go back to the PyTorch convention of channels first
        out = torch.tensor(backprop, dtype=torch.float, device=device).t()

        # better safe than sorry
        # https://discuss.pytorch.org/t/when-should-you-save-for-backward-vs-storing-in-ctx/6522/9
        # not sure this is actually necessary though
        del ctx.path
        del ctx.depth
        return out, None


def path_sig_base(path, depth):
    """See pytorch_siglayer.py's path_sig.__doc__,"""
    return path_sig_fn.apply(path, depth)
