import torch
import numpy as np


class ConvertUtt():
    def __call__(self, in_sig, batch_size):
        in_sig = in_sig.squeeze(dim=1)
        num_frames = int(in_sig.shape[0] / batch_size)
        out = in_sig.view(batch_size, num_frames, in_sig.shape[-1])

        return out


def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num


def compLossMask(inp, nframes):
    loss_mask = torch.zeros_like(inp).requires_grad_(False)
    for j, seq_len in enumerate(nframes):
        loss_mask.data[j, :, 0:seq_len] += 1.0
    return loss_mask


def snr(s, s_p):
    r""" calculate signal-to-noise ratio (SNR)

        Parameters
        ----------
        s: clean speech
        s_p: processed speech
    """
    return 10.0 * np.log10(np.sum(s ** 2) / np.sum((s_p - s) ** 2))