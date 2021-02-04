import librosa
import numpy as np
import torch
import scipy
import torch.nn as nn
import math

class ToTensor(object):
    r"""Convert ndarrays in sample to Tensors."""
    def __call__(self, x):
        return torch.from_numpy(x).float()


class SignalToFrames:
    r"""Chunks a signal into frames
         required input shape is [1, 1, -1]
         input params:    (frame_size: window_size,  frame_shift: overlap(samples))
         output:   [1, 1, num_frames, frame_size]
    """

    def __init__(self, frame_size=512, frame_shift=256):
        self.frame_size = frame_size
        self.frame_shift = frame_shift

    def __call__(self, in_sig):
        # frame_size = self.frame_size
        # frame_shift = self.frame_shift
        sig_len = in_sig.shape[-1]
        nframes = math.ceil((sig_len - self.frame_size) / self.frame_shift + 1)
        #nframes = (sig_len // (self.frame_size - self.frame_shift))
        a = np.zeros(list(in_sig.shape[:-1]) + [nframes, self.frame_size])
        start = 0
        end = start + self.frame_size
        k = 0
        for i in range(nframes):
            if end < sig_len:
                a[..., i, :] = in_sig[..., start:end]
                k += 1
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size] = in_sig[..., start:]

            start = start + self.frame_shift
            end = start + self.frame_size
        return a


class TorchSignalToFrames(object):
    """
    it is for torch tensor
    """
    def __init__(self, frame_size=512, frame_shift=256):
        super(TorchSignalToFrames, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift

    def __call__(self, in_sig):
        sig_len = in_sig.shape[-1]
        nframes = math.ceil((sig_len - self.frame_size) / self.frame_shift + 1)
        a = torch.zeros(tuple(in_sig.shape[:-1]) + (nframes, self.frame_size), device=in_sig.device)
        start = 0
        end = start + self.frame_size
        k = 0
        for i in range(nframes):
            if end < sig_len:
                a[..., i, :] = in_sig[..., start:end]
                k += 1
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size] = in_sig[..., start:]

            start = start + self.frame_shift
            end = start + self.frame_size
        return a


class OLA:
    r"""Performs overlap-and-add
        required input is ndarray
        performs frames into signal
    """
    def __init__(self, frame_shift=256):
        self.frame_shift = frame_shift

    def __call__(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = np.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype)
        ones = np.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones


class TorchOLA(nn.Module):
    r"""Overlap and add on gpu using torch tensor
        required input is tensor
        perform frames into signal
        used in the output of network
    """
    # Expects signal at last dimension
    def __init__(self, frame_shift=256):
        super(TorchOLA, self).__init__()
        self.frame_shift = frame_shift

    def forward(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = torch.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype, device=inputs.device, requires_grad=False)
        ones = torch.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones


class STFT:
    r"""Computes STFT of a signal
    input is ndarray
    required input shape is [1, 1, -1]
    """
    def __init__(self, frame_size=512, frame_shift=256):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.win = scipy.hamming(frame_size)
        self.get_frames = SignalToFrames(self.frame_size, self.frame_shift)
    def __call__(self, signal):
        frames = self.get_frames(signal)
        frames = frames*self.win
        feature = np.fft.fft(frames)[..., 0:(self.frame_size//2+1)]
        feat_R = np.real(feature)
        feat_I = np.imag(feature)
        feature = np.stack([feat_R, feat_I], axis=0)
        return feature


class ISTFT:
    r"""Computes inverse STFT"""
    # includes overlap-and-add
    def __init__(self, frame_size=512, frame_shift=256):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.win = scipy.hamming(frame_size)
        self.ola = OLA(self.frame_shift)
    def __call__(self, stft):
        R = stft[0:1, ...]
        I = stft[1:2, ...]
        cstft = R + 1j*I
        fullFFT = np.concatenate((cstft, np.conj(cstft[..., -2:0:-1])), axis=-1)
        T = np.fft.ifft(fullFFT)
        T = np.real(T)
        T = T / self.win
        signal = self.ola(T)
        return signal.astype(np.float32)


'''
# test stft and istft
path = '/data/KaiWang/pytorch_learn/pytorch_for_speech/dataset/dataset_28spk/clean_testset_wav/p232_001.wav'
wav, sr = librosa.load(path, sr=16000)
wav = np.reshape(wav, [1, 1, -1])
print(wav.shape)

get_sfft = STFT(frame_size=512, frame_shift=256)
returned = get_sfft(wav)
get_istft = ISTFT(frame_size=512, frame_shift=256)
returned1 = get_istft(returned)
print(returned1)
'''

'''
# test the overlap function 
path = '/data/KaiWang/pytorch_learn/pytorch_for_speech/dataset/dataset_28spk/clean_testset_wav/p232_001.wav'
wav, sr = librosa.load(path, sr=16000)
wav = np.reshape(wav, [1, 1, -1])
print(wav.shape)

window_size = 512
hop = int(window_size * 0.5)
get_frames = SignalToFrames(frame_size=window_size, frame_shift=hop)
returned = get_frames(wav)
print(returned)

ola = OLA(hop)
returned1 = ola(returned)
print(returned1)
'''

class SliceSig:
    '''
    input 已经是读到的audio numpy值 -- ndarray
    返回的是 frame list
    input: frame_size, hop
    '''
    def __init__(self, frame_size, hop):
        self.frame_size = frame_size
        self.frame_shift = hop

    def __call__(self, in_sig):
        sig_len = in_sig.shape[-1]
        #num_frames = math.ceil((sig_len - self.frame_shift) / (self.frame_size - self.frame_shift))
        #exp_len = num_frames * self.frame_size - (num_frames - 1) * self.frame_shift
        #pad_0 = np.zeros(exp_len - sig_len, dtype='float32')
        #pad_sig = np.concatenate((in_sig, pad_0))
        slices = []
        for end_idx in range(self.frame_size, sig_len, self.frame_size-self.frame_shift):
            start_idx = end_idx - self.frame_size
            slice_sig = in_sig[start_idx:end_idx]
            slices.append(slice_sig)
            final_idx = end_idx
        slices.append(in_sig[final_idx:])

        return slices