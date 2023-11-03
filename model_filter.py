import numpy as np
import torch
import torch.nn as nn

def lowpass_filter(audio,fs,cut_off_upper, cut_off_lower=0):
    # audio is data after reading in using tools like torchaudio.load or scipy.io.wavefile
    # fs is sample rate
    # cut_off_upper is upper limit of keep range
    # cut_off_lower is lower limit of keep range
    # work on single audio each time

    n = len(audio)  
    dt = 1/fs  
    y = np.reshape(audio,(len(audio,)))
    yf = np.fft.fft(y)/(n/2)
    freq = np.fft.fftfreq(n, dt)
    yf[(freq > cut_off_upper)] = 0
    yf[(freq < cut_off_lower)] = 0
    y = np.real(np.fft.ifft(yf)*n)
    return  y.astype("float32")

# def Xpass_filter(audio, fs, cut_off_upper, cut_off_lower=0):
#     # audio is a PyTorch tensor
#     # fs is the sample rate
#     # cut_off_upper is the upper limit of the keep range
#     # cut_off_lower is the lower limit of the keep range

#     n = audio.size(0)
#     dt = 1 / fs
#     yf = torch.fft.rfft(audio)
#     freq = torch.fft.fftfreq(n, dt)
    
#     # Apply the low-pass filter
#     yf[(freq > cut_off_upper)] = 0
#     yf[(freq < cut_off_lower)] = 0
    
#     y = torch.fft.irfft(yf, n)
#     return y.float()


class XpassFilter(nn.Module):
    def __init__(self, cut_off_upper, cut_off_lower=0):
        super(XpassFilter, self).__init__()
        self.cut_off_upper = cut_off_upper
        self.cut_off_lower = cut_off_lower

    def forward(self, audio, sample_rate):
        # audio is a PyTorch tensor with shape (num_channels, num_samples)
        num_channels, num_samples = audio.size()
        dt = 1 / sample_rate

        yf = torch.fft.rfft(audio, num_samples)
        freq = torch.fft.rfftfreq(num_samples, dt)

        # Create masks for the upper and lower cutoff frequencies
        upper_mask = (freq > self.cut_off_upper).float()
        lower_mask = (freq < self.cut_off_lower).float()

        # Apply the low-pass filter to both channels
        yf[:, :] *= (1 - upper_mask) * (1 - lower_mask)

        y = torch.fft.irfft(yf, num_samples)
        return y.float()