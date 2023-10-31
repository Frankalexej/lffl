import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
import pickle

from misc_tools import AudioCut

class DS_Tools:
    @ staticmethod
    def save_indices(filename, my_list):
        try:
            with open(filename, 'wb') as file:
                pickle.dump(my_list, file)
            return True
        except Exception as e:
            print(f"An error occurred while saving the list: {e}")
            return False

    @ staticmethod    
    def read_indices(filename):
        try:
            with open(filename, 'rb') as file:
                my_list = pickle.load(file)
            return my_list
        except Exception as e:
            print(f"An error occurred while reading the list: {e}")
            return None

class SingleRecDataset(Dataset): 
    def __init__(self, src_dir, guide_, transform=None): 
        guide_file = pd.read_csv(guide_)

        guide_file = guide_file[~guide_file["segment_nostress"].isin(["sil", "sp", "spn"])]
        guide_file = guide_file[guide_file['nSample'] > 400]
        guide_file = guide_file[guide_file['nSample'] <= 8000]

        path_col = guide_file.apply(AudioCut.record2filepath, axis=1)
        seg_col = guide_file["segment_nostress"]

        self.dataset = path_col.tolist()
        self.seg_set = seg_col.tolist()
        self.src_dir = src_dir
        self.transform = transform

    def __len__(self): 
        return len(self.dataset)

    def __getitem__(self, idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = os.path.join(
            self.src_dir, 
            self.dataset[idx]
        )

        data, sample_rate = torchaudio.load(file_name, normalize=True)
        if self.transform: 
            data = self.transform(data)
        seg = self.seg_set[idx]

        return data, seg

    @staticmethod
    def collate_fn(xx, seg):
        # only working for one data at the moment
        batch_first = True
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return (xx_pad, x_lens), seg


class MelSpecTransform(nn.Module): 
    def __init__(self, sample_rate, n_fft=400, n_mels=64): 
        super().__init__()
        self.sample_rate = sample_rate
        n_stft = int((n_fft//2) + 1)
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=n_mels, n_fft=n_fft)
        self.inverse_mel = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=n_stft)
        self.grifflim = torchaudio.transforms.GriffinLim(n_fft=n_fft)
        
    
    def forward(self, waveform): 
        # transform to mel_spectrogram
        mel_spec = self.transform(waveform)  # (channel, n_mels, time)
        mel_spec = mel_spec.squeeze()
        mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)

        """
        There should be normalization method here, 
        but for the moment we just leave it here, 
        later, consider PCEN
        """
        # # Apply normalization (CMVN)
        eps = 1e-9
        mean = mel_spec.mean(0, keepdim=True)
        std = mel_spec.std(0, keepdim=True, unbiased=False)
        mel_spec = (mel_spec - mean) / (std + eps)
        return mel_spec
    
    def de_norm(self, this_mel_spec, waveform): 
        # transform to mel_spectrogram
        mel_spec = self.transform(waveform)  # (channel, n_mels, time)
        mel_spec = mel_spec.squeeze()
        mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)

        eps = 1e-9
        mean = mel_spec.mean(0, keepdim=True)
        std = mel_spec.std(0, keepdim=True, unbiased=False)

        this_mel_spec = this_mel_spec * std + mean
        return this_mel_spec
    
    def inverse(self, mel_spec): 
        mel_spec = mel_spec.permute(1, 0) # (L, F) -> (F, L)
        mel_spec = mel_spec.unsqueeze(0)  # restore from (F, L) to (channel, F, L)
        i_mel = self.inverse_mel(mel_spec)
        inv = self.grifflim(i_mel)
        return inv