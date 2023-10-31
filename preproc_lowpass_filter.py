import numpy as np

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