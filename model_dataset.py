import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
import pickle
import random

from misc_tools import AudioCut
from model_filter import XpassFilter

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
        
class TokenMap: 
    def __init__(self, token_list):  
        self.token2idx = {element: index for index, element in enumerate(token_list)}
        self.idx2token = {index: element for index, element in enumerate(token_list)}
    
    def encode(self, token): 
        return self.token2idx[token]
    
    def decode(self, idx): 
        return self.idx2token[idx]
    
    def token_num(self): 
        return len(self.token2idx)


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
        self.mapper = TokenMap(sorted(seg_col.unique().tolist()))
        

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

        return data, self.mapper.encode(seg)

    @staticmethod
    def collate_fn(data):
        # only working for one data at the moment
        xx, seg = zip(*data)
        batch_first = True
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return (xx_pad, x_lens), seg

class SingleRecStressDataset(Dataset): 
    def __init__(self, src_dir, guide_, transform=None): 
        guide_file = pd.read_csv(guide_)

        guide_file = guide_file[~guide_file["stress_type"].isin(["SNA", "2"])]
        guide_file = guide_file[guide_file['nSample'] > 400]
        guide_file = guide_file[guide_file['nSample'] <= 8000]

        path_col = guide_file.apply(AudioCut.record2filepath, axis=1)
        seg_col = guide_file["stress_type"]

        self.dataset = path_col.tolist()
        self.seg_set = seg_col.tolist()
        self.src_dir = src_dir
        self.transform = transform
        self.mapper = TokenMap(['0', '1'])
        

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

        return data, self.mapper.encode(seg)

    @staticmethod
    def collate_fn(data):
        # only working for one data at the moment
        xx, seg = zip(*data)
        batch_first = True
        x_lens = [len(x) for x in xx]
        xx_pad = pad_sequence(xx, batch_first=batch_first, padding_value=0)
        return (xx_pad, x_lens), seg
    

class PairRecDataset():
    def __init__(self, aud_dir, log_file, total_number=1000, transform=None):
        # we define a total number so that we can freely adjust the total number of training examples we take hold of. 
        # Because we definitely don't use up the whole pair set. 
        self.log = pd.read_csv(log_file)
        self.aud_dir = aud_dir
        self.total_number = total_number
        self.transform = transform

        # group audios based on label
        self.group_examples()

    def group_examples(self):
        """
            To ease the accessibility of data based on the class, we will use `group_examples` to group
            examples based on label.

            Every key in `grouped_examples` corresponds to a label in the dataset. For every key in
            `grouped_examples`, every value will conform to all of the indices for the
            audio that correspond to that label.
        """

        # this will return a dictionary of Index class objects, which I think is similar to list
        self.grouped_examples = self.log.groupby('segment_nostress').groups

        # self.grouped_examples = {}

        # for i, row in self.log.iterrows():
        #     label = row['segment']
        #     if label not in self.grouped_examples:
        #         self.grouped_examples[label] = []
        #     self.grouped_examples[label].append(i)

    def __len__(self):
        return self.total_number

    def __getitem__(self, index):
        """
            For every example, we will select two images. There are two cases,
            positive and negative examples. For positive examples, we will have two
            images from the same class. For negative examples, we will have two images
            from different classes.

            Given an index, if the index is even, we will pick the second image from the same class,
            and it may be same image we chose for the first class. If the index is odd, we will
            pick the second image from a different class than the first image.
        """

        # pick a random label for the first sample
        selected_label = random.choice(list(self.grouped_examples.keys()))

        # pick a random index for the first sample in the grouped indices based on the label
        selected_index_1 = random.choice(self.grouped_examples[selected_label])

        # get the first sample
        folders_1 = self.log.loc[selected_index_1, 'file']
        id_1 = self.log.loc[selected_index_1, 'id']
        # file_1 = str(folders_1[0]) + "-" + str(folders_1[1]) + "-" + str(folders_1[2]) + "-" + str(id_1) + ".flac"
        path_1 = os.path.join(self.aud_dir, AudioCut.filename_id2filepath(folders_1, id_1))
        audio_1, sample_rate = torchaudio.load(path_1, normalize=True)

        # same class
        if index % 2 == 0:
            # pick a random index for the second sample
            selected_index_2 = random.choice(self.grouped_examples[selected_label])

            # get the second sample
            folders_2 = self.log.loc[selected_index_2, 'file']
            id_2 = self.log.loc[selected_index_2, 'id']
            # file_2 = str(folders_2[0]) + "-" + str(folders_2[1]) + "-" + str(folders_2[2]) + "-" + str(id_2) + ".flac"
            path_2 = os.path.join(self.aud_dir, AudioCut.filename_id2filepath(folders_2, id_2))
            audio_2, sample_rate = torchaudio.load(path_2, normalize=True)

            # set the label for this example to be positive (1)
            # target = torch.tensor(1, dtype=torch.float)
            target = 1.

        # different class
        else:
            # pick a random label
            other_selected_label = random.choice(list(self.grouped_examples.keys()))

            # ensure that the label of the second sample isn't the same as the first sample
            while other_selected_label == selected_label:
                other_selected_label = random.choice(list(self.grouped_examples.keys()))

            # pick a random index for the second sample in the grouped indices based on the label
            selected_index_2 = random.choice(self.grouped_examples[selected_label])

            # get the second sample
            folders_2 = self.log.loc[selected_index_2, 'file']
            id_2 = self.log.loc[selected_index_2, 'id']
            # file_2 = str(folders_2[0]) + "-" + str(folders_2[1]) + "-" + str(folders_2[2]) + "-" + str(id_2) + ".flac"
            path_2 = os.path.join(self.aud_dir, AudioCut.filename_id2filepath(folders_2, id_2))
            audio_2, sample_rate = torchaudio.load(path_2, normalize=True)

            # set the label for this example to be negative (0)
            # target = torch.tensor(0, dtype=torch.float)
            target = 0.

        if self.transform: 
            audio_1 = self.transform(audio_1)
            audio_2 = self.transform(audio_2)

        return audio_1, audio_2, target
    
    @staticmethod
    def collate_fn(data):
        # only working for one data at the moment
        xx_1, xx_2, target = zip(*data)
        batch_first = True
        x_1_lens = [len(x) for x in xx_1]
        x_2_lens = [len(x) for x in xx_2]
        xx_1_pad = pad_sequence(xx_1, batch_first=batch_first, padding_value=0)
        xx_2_pad = pad_sequence(xx_2, batch_first=batch_first, padding_value=0)
        return (xx_1_pad, x_1_lens), (xx_2_pad, x_2_lens), target



class MelSpecTransform(nn.Module): 
    def __init__(self, sample_rate, n_fft=400, n_mels=64, filter=None): 
        super().__init__()
        self.sample_rate = sample_rate
        n_stft = int((n_fft//2) + 1)
        self.filter = filter
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=n_mels, n_fft=n_fft)
        self.inverse_mel = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=n_stft)
        self.grifflim = torchaudio.transforms.GriffinLim(n_fft=n_fft)
        
    
    def forward(self, waveform): 
        # transform to mel_spectrogram
        if self.filter: 
            waveform = self.filter(waveform, self.sample_rate)

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