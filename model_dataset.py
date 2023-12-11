import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import os
import pickle
import random
from scipy import signal
# import librosa

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

class SingleRecSelectBalanceDatasetPrecombine(Dataset): 
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None): 
        guide_file = pd.read_csv(guide_)

        guide_file = guide_file[guide_file["segment_nostress"].isin(select)]
        guide_file = guide_file[guide_file['nSample'] > 400]
        guide_file = guide_file[guide_file['nSample'] <= 8000]

        # guide_file = self.balance_dataframe(guide_file, "segment_nostress")

        # path_col = guide_file.apply(AudioCut.record2filepath, axis=1)
        path_col = guide_file["combined_path"]
        seg_col = guide_file["segment_nostress"]

        self.dataset = path_col.tolist()
        self.seg_set = seg_col.tolist()
        self.src_dir = src_dir
        self.transform = transform
        if mapper: 
            self.mapper = mapper
        else: 
            self.mapper = TokenMap(sorted(seg_col.unique().tolist()))
        
        self.mapper = TokenMap(select)
        

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
    def balance_dataframe(df, tag_column):
        """
        Balance the dataset in a Pandas DataFrame by randomly reducing the number of samples for each tag to match the minimum number of samples.

        Parameters:
        - df: Pandas DataFrame containing audio samples and corresponding tags.
        - audio_column: Name of the column containing audio samples in the DataFrame.
        - tag_column: Name of the column containing tags in the DataFrame.

        Returns:
        - balanced_df: Pandas DataFrame with a balanced number of samples for each tag.
        """
        # Find the minimum number of samples for any tag
        min_samples = min(df[tag_column].value_counts())

        # Create a dictionary to store indices of samples for each tag
        tag_indices = {tag: df.index[df[tag_column] == tag].tolist() for tag in df[tag_column].unique()}

        # Randomly select indices to keep for each tag to balance the dataset
        balanced_indices = []
        for tag, indices in tag_indices.items():
            balanced_indices.extend(random.sample(indices, min_samples))

        # Create balanced DataFrame using the selected indices
        balanced_df = df.loc[balanced_indices].reset_index(drop=True)

        return balanced_df
    

class SingleRecSmallDatasetPrecombine(Dataset): 
    def __init__(self, src_dir, guide_, select=[], mapper=None, transform=None): 
        guide_file = pd.read_csv(guide_)

        guide_file = guide_file[guide_file["segment_nostress"].isin(select)]
        guide_file = guide_file[guide_file['nSample'] > 400]
        guide_file = guide_file[guide_file['nSample'] <= 8000]

        # path_col = guide_file.apply(AudioCut.record2filepath, axis=1)
        path_col = guide_file["combined_path"]
        seg_col = guide_file["segment_nostress"]

        self.dataset = path_col.tolist()
        self.seg_set = seg_col.tolist()
        self.src_dir = src_dir
        self.transform = transform
        if mapper: 
            self.mapper = mapper
        else: 
            self.mapper = TokenMap(sorted(seg_col.unique().tolist()))
        
        self.mapper = TokenMap(select)
        

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
    
class SingleRecNoAHDatasetPrecombine(Dataset): 
    def __init__(self, src_dir, guide_, mapper=None, transform=None): 
        guide_file = pd.read_csv(guide_)

        guide_file = guide_file[~guide_file["segment_nostress"].isin(["sil", "sp", "spn", "AH"])]
        guide_file = guide_file[guide_file['nSample'] > 400]
        guide_file = guide_file[guide_file['nSample'] <= 8000]

        # path_col = guide_file.apply(AudioCut.record2filepath, axis=1)
        path_col = guide_file["combined_path"]
        seg_col = guide_file["segment_nostress"]

        self.dataset = path_col.tolist()
        self.seg_set = seg_col.tolist()
        self.src_dir = src_dir
        self.transform = transform
        if mapper: 
            self.mapper = mapper
        else: 
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


class SingleRecNoAHDataset(Dataset): 
    def __init__(self, src_dir, guide_, mapper=None, transform=None): 
        guide_file = pd.read_csv(guide_)

        guide_file = guide_file[~guide_file["segment_nostress"].isin(["sil", "sp", "spn", "AH"])]
        guide_file = guide_file[guide_file['nSample'] > 400]
        guide_file = guide_file[guide_file['nSample'] <= 8000]

        path_col = guide_file.apply(AudioCut.record2filepath, axis=1)
        seg_col = guide_file["segment_nostress"]

        self.dataset = path_col.tolist()
        self.seg_set = seg_col.tolist()
        self.src_dir = src_dir
        self.transform = transform
        if mapper: 
            self.mapper = mapper
        else: 
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

class SingleRecDataset(Dataset): 
    def __init__(self, src_dir, guide_, mapper=None, transform=None): 
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
        if mapper: 
            self.mapper = mapper
        else: 
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
    

class PairRecDataset(Dataset):
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
            target = 0.

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
            target = 1.

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

class PairRecDatasetPregen(Dataset):
    
    def __init__(self, aud_dir, log_file, list_save_dir, total_number=10000, transform=None):
        # we define a total number so that we can freely adjust the total number of training examples we take hold of.
        # Because we definitely don't use up the whole pair set.
        self.aud_dir = aud_dir
        self.list_save_dir = list_save_dir
        self.log = pd.read_csv(log_file)
        self.total_number = total_number
        self.transform = transform

        # group audios based on label
        self.group_examples()

        # initialize the list of paired audio
        self.get_list()

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
        #
        # for i, row in self.log.iterrows():
        #     label = row['segment']
        #     if label not in self.grouped_examples:
        #         self.grouped_examples[label] = []
        #     self.grouped_examples[label].append(i)

    def get_pair(self, index):
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
        # audio_1, sample_rate = torchaudio.load(path_1, normalize=True)

        # same class
        if index % 2 == 0:
            # pick a random index for the second sample
            selected_index_2 = random.choice(self.grouped_examples[selected_label])

            # get the second sample
            folders_2 = self.log.loc[selected_index_2, 'file']
            id_2 = self.log.loc[selected_index_2, 'id']
            # file_2 = str(folders_2[0]) + "-" + str(folders_2[1]) + "-" + str(folders_2[2]) + "-" + str(id_2) + ".flac"
            path_2 = os.path.join(self.aud_dir, AudioCut.filename_id2filepath(folders_2, id_2))
            # audio_2, sample_rate = torchaudio.load(path_2, normalize=True)

            # set the label for this example to be positive (1)
            # target = torch.tensor(1, dtype=torch.float)
            target = 0.

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
            # audio_2, sample_rate = torchaudio.load(path_2, normalize=True)

            # set the label for this example to be negative (0)
            # target = torch.tensor(0, dtype=torch.float)
            target = 1.

        return path_1, path_2, target

    def get_list(self):
        """
        Generate a list of paired audio samples.

        The list will contain `total_number` pairs of audio samples, where each pair consists of two audio samples
        and a target label indicating whether the samples are from the same class (0) or different classes (1).
        """
        # a dataframe where each entry is a pair of audio
        self.paired_audio = pd.DataFrame(columns=["path_1", "path_2", "target"])

        # get pairs for total_number of times and append to dataframe
        for i in range(self.total_number):
            new_line = pd.Series(self.get_pair(i), index=self.paired_audio.columns)
            self.paired_audio = self.paired_audio._append(new_line, ignore_index=True)

        # write to csv
        self.paired_audio.to_csv(os.path.join(self.list_save_dir, "paired_dataset.csv"), index=False)

    def __len__(self):
        return self.total_number

    def __getitem__(self, index):
        path_1 = self.paired_audio.iloc[index, 0]
        audio_1, sample_rate = torchaudio.load(path_1, normalize=True)
        path_2 = self.paired_audio.iloc[index, 1]
        audio_2, sample_rate = torchaudio.load(path_2, normalize=True)
        target = self.paired_audio.iloc[index, 2]

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

class PairRecDatasetPregenPrecombine(Dataset):
    
    def __init__(self, aud_dir, log_file, list_save_dir, total_number=10000, transform=None, whether_load_from=False, load_from=None):
        # we define a total number so that we can freely adjust the total number of training examples we take hold of.
        # Because we definitely don't use up the whole pair set.
        self.aud_dir = aud_dir
        self.list_save_dir = list_save_dir
        self.total_number = total_number
        self.transform = transform
        if (not whether_load_from) or (not load_from): 
            self.log = pd.read_csv(log_file)
            # group audios based on label
            self.group_examples()
            # initialize the list of paired audio
            self.get_list()
        else: 
            self.paired_audio = pd.read_csv(load_from)
        

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
        #
        # for i, row in self.log.iterrows():
        #     label = row['segment']
        #     if label not in self.grouped_examples:
        #         self.grouped_examples[label] = []
        #     self.grouped_examples[label].append(i)

    def get_pair(self, index):
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
        # folders_1 = self.log.loc[selected_index_1, 'file']
        # id_1 = self.log.loc[selected_index_1, 'id']
        # file_1 = str(folders_1[0]) + "-" + str(folders_1[1]) + "-" + str(folders_1[2]) + "-" + str(id_1) + ".flac"
        # path_1 = os.path.join(self.aud_dir, AudioCut.filename_id2filepath(folders_1, id_1))
        path_1 = os.path.join(self.aud_dir, self.log.loc[selected_index_1, 'combined_path'])
        # audio_1, sample_rate = torchaudio.load(path_1, normalize=True)

        # same class
        if index % 2 == 0:
            # pick a random index for the second sample
            selected_index_2 = random.choice(self.grouped_examples[selected_label])

            # get the second sample
            # folders_2 = self.log.loc[selected_index_2, 'file']
            # id_2 = self.log.loc[selected_index_2, 'id']
            # file_2 = str(folders_2[0]) + "-" + str(folders_2[1]) + "-" + str(folders_2[2]) + "-" + str(id_2) + ".flac"
            # path_2 = os.path.join(self.aud_dir, AudioCut.filename_id2filepath(folders_2, id_2))
            path_2 = os.path.join(self.aud_dir, self.log.loc[selected_index_2, 'combined_path'])
            # audio_2, sample_rate = torchaudio.load(path_2, normalize=True)

            # set the label for this example to be positive (1)
            # target = torch.tensor(1, dtype=torch.float)
            target = 0.

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
            # folders_2 = self.log.loc[selected_index_2, 'file']
            # id_2 = self.log.loc[selected_index_2, 'id']
            # file_2 = str(folders_2[0]) + "-" + str(folders_2[1]) + "-" + str(folders_2[2]) + "-" + str(id_2) + ".flac"
            # path_2 = os.path.join(self.aud_dir, AudioCut.filename_id2filepath(folders_2, id_2))
            path_2 = os.path.join(self.aud_dir, self.log.loc[selected_index_2, 'combined_path'])
            # audio_2, sample_rate = torchaudio.load(path_2, normalize=True)

            # set the label for this example to be negative (0)
            # target = torch.tensor(0, dtype=torch.float)
            target = 1.

        return path_1, path_2, target

    def get_list(self):
        """
        Generate a list of paired audio samples.

        The list will contain `total_number` pairs of audio samples, where each pair consists of two audio samples
        and a target label indicating whether the samples are from the same class (0) or different classes (1).
        """
        # a dataframe where each entry is a pair of audio
        self.paired_audio = pd.DataFrame(columns=["path_1", "path_2", "target"])

        # get pairs for total_number of times and append to dataframe
        for i in range(self.total_number):
            new_line = pd.Series(self.get_pair(i), index=self.paired_audio.columns)
            self.paired_audio = self.paired_audio._append(new_line, ignore_index=True)

        # write to csv
        self.paired_audio.to_csv(self.list_save_dir, index=False)

    def __len__(self):
        return self.total_number

    def __getitem__(self, index):
        path_1 = self.paired_audio.iloc[index, 0]
        audio_1, sample_rate = torchaudio.load(path_1, normalize=True)
        path_2 = self.paired_audio.iloc[index, 1]
        audio_2, sample_rate = torchaudio.load(path_2, normalize=True)
        target = self.paired_audio.iloc[index, 2]

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
    
class PairRecDatasetPregenPrecombineNoAH(Dataset):
    
    def __init__(self, aud_dir, log_file, list_save_dir, total_number=10000, transform=None, whether_load_from=False, load_from=None):
        # we define a total number so that we can freely adjust the total number of training examples we take hold of.
        # Because we definitely don't use up the whole pair set.
        self.aud_dir = aud_dir
        self.list_save_dir = list_save_dir
        self.total_number = total_number
        self.transform = transform
        if (not whether_load_from) or (not load_from): 
            self.log = pd.read_csv(log_file)

            self.log = self.log[~self.log["segment_nostress"].isin(["sil", "sp", "spn", "AH"])]
            self.log = self.log[self.log['nSample'] > 400]
            self.log = self.log[self.log['nSample'] <= 8000]

            # group audios based on label
            self.group_examples()
            # initialize the list of paired audio
            self.get_list()
        else: 
            self.paired_audio = pd.read_csv(load_from)
        

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
        #
        # for i, row in self.log.iterrows():
        #     label = row['segment']
        #     if label not in self.grouped_examples:
        #         self.grouped_examples[label] = []
        #     self.grouped_examples[label].append(i)

    def get_pair(self, index):
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
        # folders_1 = self.log.loc[selected_index_1, 'file']
        # id_1 = self.log.loc[selected_index_1, 'id']
        # file_1 = str(folders_1[0]) + "-" + str(folders_1[1]) + "-" + str(folders_1[2]) + "-" + str(id_1) + ".flac"
        # path_1 = os.path.join(self.aud_dir, AudioCut.filename_id2filepath(folders_1, id_1))
        path_1 = os.path.join(self.aud_dir, self.log.loc[selected_index_1, 'combined_path'])
        # audio_1, sample_rate = torchaudio.load(path_1, normalize=True)

        # same class
        if index % 2 == 0:
            # pick a random index for the second sample
            selected_index_2 = random.choice(self.grouped_examples[selected_label])

            # get the second sample
            # folders_2 = self.log.loc[selected_index_2, 'file']
            # id_2 = self.log.loc[selected_index_2, 'id']
            # file_2 = str(folders_2[0]) + "-" + str(folders_2[1]) + "-" + str(folders_2[2]) + "-" + str(id_2) + ".flac"
            # path_2 = os.path.join(self.aud_dir, AudioCut.filename_id2filepath(folders_2, id_2))
            path_2 = os.path.join(self.aud_dir, self.log.loc[selected_index_2, 'combined_path'])
            # audio_2, sample_rate = torchaudio.load(path_2, normalize=True)

            # set the label for this example to be positive (1)
            # target = torch.tensor(1, dtype=torch.float)
            target = 0.

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
            # folders_2 = self.log.loc[selected_index_2, 'file']
            # id_2 = self.log.loc[selected_index_2, 'id']
            # file_2 = str(folders_2[0]) + "-" + str(folders_2[1]) + "-" + str(folders_2[2]) + "-" + str(id_2) + ".flac"
            # path_2 = os.path.join(self.aud_dir, AudioCut.filename_id2filepath(folders_2, id_2))
            path_2 = os.path.join(self.aud_dir, self.log.loc[selected_index_2, 'combined_path'])
            # audio_2, sample_rate = torchaudio.load(path_2, normalize=True)

            # set the label for this example to be negative (0)
            # target = torch.tensor(0, dtype=torch.float)
            target = 1.

        return path_1, path_2, target

    def get_list(self):
        """
        Generate a list of paired audio samples.

        The list will contain `total_number` pairs of audio samples, where each pair consists of two audio samples
        and a target label indicating whether the samples are from the same class (0) or different classes (1).
        """
        # a dataframe where each entry is a pair of audio
        self.paired_audio = pd.DataFrame(columns=["path_1", "path_2", "target"])

        # get pairs for total_number of times and append to dataframe
        for i in range(self.total_number):
            new_line = pd.Series(self.get_pair(i), index=self.paired_audio.columns)
            self.paired_audio = self.paired_audio._append(new_line, ignore_index=True)

        # write to csv
        self.paired_audio.to_csv(self.list_save_dir, index=False)

    def __len__(self):
        return self.total_number

    def __getitem__(self, index):
        path_1 = self.paired_audio.iloc[index, 0]
        audio_1, sample_rate = torchaudio.load(path_1, normalize=True)
        path_2 = self.paired_audio.iloc[index, 1]
        audio_2, sample_rate = torchaudio.load(path_2, normalize=True)
        target = self.paired_audio.iloc[index, 2]

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
    
class PairRecDatasetPregenPrecombineSmall(Dataset):
    
    def __init__(self, aud_dir, log_file, list_save_dir, total_number=10000, positive_every=2, transform=None, whether_load_from=False, load_from=None):
        # we define a total number so that we can freely adjust the total number of training examples we take hold of.
        # Because we definitely don't use up the whole pair set.
        self.aud_dir = aud_dir
        self.list_save_dir = list_save_dir
        self.total_number = total_number
        self.transform = transform
        if (not whether_load_from) or (not load_from): 
            self.log = pd.read_csv(log_file)

            self.log = self.log[self.log["segment_nostress"].isin(["IY", "AE", "OW"])]
            self.log = self.log[self.log['nSample'] > 400]
            self.log = self.log[self.log['nSample'] <= 8000]

            self.positive_every = positive_every

            # group audios based on label
            self.group_examples()
            # initialize the list of paired audio
            self.get_list()
        else: 
            self.paired_audio = pd.read_csv(load_from)
        

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
        #
        # for i, row in self.log.iterrows():
        #     label = row['segment']
        #     if label not in self.grouped_examples:
        #         self.grouped_examples[label] = []
        #     self.grouped_examples[label].append(i)

    def get_pair(self, index):
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
        # folders_1 = self.log.loc[selected_index_1, 'file']
        # id_1 = self.log.loc[selected_index_1, 'id']
        # file_1 = str(folders_1[0]) + "-" + str(folders_1[1]) + "-" + str(folders_1[2]) + "-" + str(id_1) + ".flac"
        # path_1 = os.path.join(self.aud_dir, AudioCut.filename_id2filepath(folders_1, id_1))
        path_1 = os.path.join(self.aud_dir, self.log.loc[selected_index_1, 'combined_path'])
        # audio_1, sample_rate = torchaudio.load(path_1, normalize=True)

        # same class
        if index % self.positive_every == 0:
            # pick a random index for the second sample
            selected_index_2 = random.choice(self.grouped_examples[selected_label])

            # get the second sample
            # folders_2 = self.log.loc[selected_index_2, 'file']
            # id_2 = self.log.loc[selected_index_2, 'id']
            # file_2 = str(folders_2[0]) + "-" + str(folders_2[1]) + "-" + str(folders_2[2]) + "-" + str(id_2) + ".flac"
            # path_2 = os.path.join(self.aud_dir, AudioCut.filename_id2filepath(folders_2, id_2))
            path_2 = os.path.join(self.aud_dir, self.log.loc[selected_index_2, 'combined_path'])
            # audio_2, sample_rate = torchaudio.load(path_2, normalize=True)

            # set the label for this example to be positive (1)
            # target = torch.tensor(1, dtype=torch.float)
            target = 1.

            label_1 = selected_label
            label_2 = selected_label

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
            # folders_2 = self.log.loc[selected_index_2, 'file']
            # id_2 = self.log.loc[selected_index_2, 'id']
            # file_2 = str(folders_2[0]) + "-" + str(folders_2[1]) + "-" + str(folders_2[2]) + "-" + str(id_2) + ".flac"
            # path_2 = os.path.join(self.aud_dir, AudioCut.filename_id2filepath(folders_2, id_2))
            path_2 = os.path.join(self.aud_dir, self.log.loc[selected_index_2, 'combined_path'])
            # audio_2, sample_rate = torchaudio.load(path_2, normalize=True)

            # set the label for this example to be negative (0)
            # target = torch.tensor(0, dtype=torch.float)
            target = 0.

            label_1 = selected_label
            label_2 = other_selected_label

        return path_1, path_2, target, label_1, label_2

    def get_list(self):
        """
        Generate a list of paired audio samples.

        The list will contain `total_number` pairs of audio samples, where each pair consists of two audio samples
        and a target label indicating whether the samples are from the same class (0) or different classes (1).
        """
        # a dataframe where each entry is a pair of audio
        self.paired_audio = pd.DataFrame(columns=["path_1", "path_2", "target", "label_1", "label_2"])

        # get pairs for total_number of times and append to dataframe
        for i in range(self.total_number):
            new_line = pd.Series(self.get_pair(i), index=self.paired_audio.columns)
            self.paired_audio = self.paired_audio._append(new_line, ignore_index=True)

        # write to csv
        self.paired_audio.to_csv(self.list_save_dir, index=False)

    def __len__(self):
        return self.total_number

    def __getitem__(self, index):
        path_1 = self.paired_audio.iloc[index, 0]
        audio_1, sample_rate = torchaudio.load(path_1, normalize=True)
        path_2 = self.paired_audio.iloc[index, 1]
        audio_2, sample_rate = torchaudio.load(path_2, normalize=True)
        target = self.paired_audio.iloc[index, 2]

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
    
class PairRecDatasetPregenPrecombineSelect(Dataset):
    
    def __init__(self, aud_dir, log_file, list_save_dir, select=[], total_number=10000, positive_every=2, transform=None, whether_load_from=False, load_from=None):
        # we define a total number so that we can freely adjust the total number of training examples we take hold of.
        # Because we definitely don't use up the whole pair set.
        self.aud_dir = aud_dir
        self.list_save_dir = list_save_dir
        self.total_number = total_number
        self.transform = transform
        if (not whether_load_from) or (not load_from): 
            self.log = pd.read_csv(log_file)

            self.log = self.log[self.log["segment_nostress"].isin(select)]
            self.log = self.log[self.log['nSample'] > 400]
            self.log = self.log[self.log['nSample'] <= 8000]

            self.positive_every = positive_every

            # group audios based on label
            self.group_examples()
            # initialize the list of paired audio
            self.get_list()
        else: 
            self.paired_audio = pd.read_csv(load_from)
        

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
        #
        # for i, row in self.log.iterrows():
        #     label = row['segment']
        #     if label not in self.grouped_examples:
        #         self.grouped_examples[label] = []
        #     self.grouped_examples[label].append(i)

    def get_pair(self, index):
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
        # folders_1 = self.log.loc[selected_index_1, 'file']
        # id_1 = self.log.loc[selected_index_1, 'id']
        # file_1 = str(folders_1[0]) + "-" + str(folders_1[1]) + "-" + str(folders_1[2]) + "-" + str(id_1) + ".flac"
        # path_1 = os.path.join(self.aud_dir, AudioCut.filename_id2filepath(folders_1, id_1))
        path_1 = os.path.join(self.aud_dir, self.log.loc[selected_index_1, 'combined_path'])
        # audio_1, sample_rate = torchaudio.load(path_1, normalize=True)

        # same class
        if index % self.positive_every == 0:
            # pick a random index for the second sample
            selected_index_2 = random.choice(self.grouped_examples[selected_label])

            # get the second sample
            # folders_2 = self.log.loc[selected_index_2, 'file']
            # id_2 = self.log.loc[selected_index_2, 'id']
            # file_2 = str(folders_2[0]) + "-" + str(folders_2[1]) + "-" + str(folders_2[2]) + "-" + str(id_2) + ".flac"
            # path_2 = os.path.join(self.aud_dir, AudioCut.filename_id2filepath(folders_2, id_2))
            path_2 = os.path.join(self.aud_dir, self.log.loc[selected_index_2, 'combined_path'])
            # audio_2, sample_rate = torchaudio.load(path_2, normalize=True)

            # set the label for this example to be positive (1)
            # target = torch.tensor(1, dtype=torch.float)
            target = 1.

            label_1 = selected_label
            label_2 = selected_label

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
            # folders_2 = self.log.loc[selected_index_2, 'file']
            # id_2 = self.log.loc[selected_index_2, 'id']
            # file_2 = str(folders_2[0]) + "-" + str(folders_2[1]) + "-" + str(folders_2[2]) + "-" + str(id_2) + ".flac"
            # path_2 = os.path.join(self.aud_dir, AudioCut.filename_id2filepath(folders_2, id_2))
            path_2 = os.path.join(self.aud_dir, self.log.loc[selected_index_2, 'combined_path'])
            # audio_2, sample_rate = torchaudio.load(path_2, normalize=True)

            # set the label for this example to be negative (0)
            # target = torch.tensor(0, dtype=torch.float)
            target = 0.

            label_1 = selected_label
            label_2 = other_selected_label

        return path_1, path_2, target, label_1, label_2

    def get_list(self):
        """
        Generate a list of paired audio samples.

        The list will contain `total_number` pairs of audio samples, where each pair consists of two audio samples
        and a target label indicating whether the samples are from the same class (0) or different classes (1).
        """
        # a dataframe where each entry is a pair of audio
        self.paired_audio = pd.DataFrame(columns=["path_1", "path_2", "target", "label_1", "label_2"])

        # get pairs for total_number of times and append to dataframe
        for i in range(self.total_number):
            new_line = pd.Series(self.get_pair(i), index=self.paired_audio.columns)
            self.paired_audio = self.paired_audio._append(new_line, ignore_index=True)

        # write to csv
        self.paired_audio.to_csv(self.list_save_dir, index=False)

    def __len__(self):
        return self.total_number

    def __getitem__(self, index):
        path_1 = self.paired_audio.iloc[index, 0]
        audio_1, sample_rate = torchaudio.load(path_1, normalize=True)
        path_2 = self.paired_audio.iloc[index, 1]
        audio_2, sample_rate = torchaudio.load(path_2, normalize=True)
        target = self.paired_audio.iloc[index, 2]

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
    


class Resampler(nn.Module): 
    def __init__(self, target_frame_num=25, axis=0): 
        super().__init__()
        self.target_frame_num = target_frame_num
        self.axis = axis
    
    def forward(self, mfcc): 
        return torch.tensor(signal.resample(mfcc, self.target_frame_num, axis=self.axis))


class MelSpecTransform(nn.Module): 
    def __init__(self, sample_rate, n_fft=400, n_mels=64, filter=None, norm=None, resample=None): 
        super().__init__()
        self.sample_rate = sample_rate
        n_stft = int((n_fft//2) + 1)
        self.filter = filter
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=n_mels, n_fft=n_fft, power=2)
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        self.inverse_mel = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=n_stft)
        self.grifflim = torchaudio.transforms.GriffinLim(n_fft=n_fft)
        self.norm = norm
        self.resample = resample
    
    def forward(self, waveform):
        # transform to mel_spectrogram
        if self.filter: 
            waveform = self.filter(waveform, self.sample_rate)

        mel_spec = self.transform(waveform)  # (channel, n_mels, time)
        mel_spec = mel_spec.squeeze()
        mel_spec = mel_spec.permute(1, 0) # (F, L) -> (L, F)

        # mel_spec = torch.log(mel_spec + 1e-9)   # 20231121 newly added log
        mel_spec = self.amp_to_db(mel_spec)
        if self.norm: 
            mel_spec = self.norm(mel_spec)
        
        if self.resample: 
            mel_spec = self.resample(mel_spec)

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
    

class SpecTransform(nn.Module): 
    def __init__(self, sample_rate=16000, n_fft=400, filter=None): 
        super().__init__()
        self.sample_rate = sample_rate
        self.filter = filter
        self.transform = torchaudio.transforms.Spectrogram(n_fft=n_fft)
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        
    
    def forward(self, waveform): 
        # transform to mel_spectrogram
        if self.filter: 
            waveform = self.filter(waveform, self.sample_rate)

        spec = self.transform(waveform)  # (channel, n_mels, time)
        spec = spec.squeeze()
        spec = spec.permute(1, 0) # (F, L) -> (L, F)
        spec = self.amp_to_db(spec)
        spec = self.norm_minmax(spec)
        return spec
    
    def norm_mvn(self, mel_spec):
        eps = 1e-9
        mean = mel_spec.mean(0, keepdim=True)
        std = mel_spec.std(0, keepdim=True, unbiased=False)
        norm_spec = (mel_spec - mean) / (std + eps)
        return norm_spec
    
    def norm_minmax(self, mel_spec):
        min_val = mel_spec.min(0)[1]
        max_val = mel_spec.max(0)[1]
        norm_spec = (mel_spec - min_val) / (max_val - min_val)
        return norm_spec
    
    def norm_pcen(self, mel_spec):
        pass
        # librosa.pcen(mel_spec[0].detach()[0,:,:].numpy().copy()*(2**31), sr=self.sample_rate)

class Normalizer(nn.Module): 
    def __init__(self, fun):
        super().__init__()
        self.fun = fun
    
    def forward(self, mel_spec):
        return self.fun(mel_spec)
    
    @staticmethod
    def norm_strip_mvn(mel_spec):
        eps = 1e-9
        mean = mel_spec.mean(1, keepdim=True)
        std = mel_spec.std(1, keepdim=True, unbiased=False)
        norm_spec = (mel_spec - mean) / (std + eps)
        return norm_spec

    @staticmethod
    def norm_mvn(mel_spec):
        eps = 1e-9
        mean = mel_spec.mean()
        std = mel_spec.std(unbiased=False)
        norm_spec = (mel_spec - mean) / (std + eps)
        return norm_spec
    
    @staticmethod
    def norm_time_mvn(mel_spec):
        eps = 1e-9
        mean = mel_spec.mean(0, keepdim=True)
        std = mel_spec.std(0, keepdim=True, unbiased=False)
        norm_spec = (mel_spec - mean) / (std + eps)
        return norm_spec

    @staticmethod
    def norm_minmax(mel_spec):
        min_val = mel_spec.min()
        max_val = mel_spec.max()
        norm_spec = (mel_spec - min_val) / (max_val - min_val)
        return norm_spec
    
    @staticmethod
    def norm_strip_minmax(mel_spec):
        min_val = mel_spec.min(1, keepdim=True)[0]
        max_val = mel_spec.max(1, keepdim=True)[0]
        norm_spec = (mel_spec - min_val) / (max_val - min_val)
        return norm_spec
    
    @staticmethod
    def norm_pcen(mel_spec):
        return mel_spec
    

class NormalizerKeepShape(nn.Module):
    # this normalizer will work on mels that keep the shape as (channel, F, L) 
    # strip: normalize and reduce along the frequency axis
    # time: normalize and reduce along the time axis
    # -: global normalization across whole mel
    def __init__(self, fun):
        super().__init__()
        self.fun = fun
    
    def forward(self, mel_spec):
        return self.fun(mel_spec)

    @staticmethod
    def norm_strip_mvn(mel_spec):
        eps = 1e-9
        mean = mel_spec.mean(1, keepdim=True)
        std = mel_spec.std(1, keepdim=True, unbiased=False)
        norm_spec = (mel_spec - mean) / (std + eps)
        return norm_spec

    @staticmethod
    def norm_mvn(mel_spec):
        eps = 1e-9
        mean = mel_spec.mean()
        std = mel_spec.std(unbiased=False)
        norm_spec = (mel_spec - mean) / (std + eps)
        return norm_spec
    
    @staticmethod
    def norm_time_mvn(mel_spec):
        eps = 1e-9
        mean = mel_spec.mean(2, keepdim=True)
        std = mel_spec.std(2, keepdim=True, unbiased=False)
        norm_spec = (mel_spec - mean) / (std + eps)
        return norm_spec

    @staticmethod
    def norm_minmax(mel_spec):
        min_val = mel_spec.min()
        max_val = mel_spec.max()
        norm_spec = (mel_spec - min_val) / (max_val - min_val)
        return norm_spec
    
    @staticmethod
    def norm_strip_minmax(mel_spec):
        min_val = mel_spec.min(1, keepdim=True)[0]
        max_val = mel_spec.max(1, keepdim=True)[0]
        norm_spec = (mel_spec - min_val) / (max_val - min_val)
        return norm_spec
    
    @staticmethod
    def norm_pcen(mel_spec):
        return mel_spec
    
class Padder(nn.Module): 
    def __init__(self, sample_rate=16000, pad_len_ms=250, noise_level=0.001): 
        super().__init__()
        self.sample_rate = sample_rate
        self.pad_len_frame = sample_rate // 1000 * pad_len_ms
        self.noise_level = noise_level
    
    def forward(self, sig): 
        num_rows, sig_len = sig.shape
        if (sig_len > self.pad_len_frame):
            # Truncate the signal to the given length
            sig = sig[:,:self.pad_len_frame]

        elif (sig_len < self.pad_len_frame):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, self.pad_len_frame - sig_len)
            pad_end_len = self.pad_len_frame - sig_len - pad_begin_len

            # Pad with 0s
            # pad_begin = torch.zeros((num_rows, pad_begin_len))
            # pad_end = torch.zeros((num_rows, pad_end_len))
            pad_begin = torch.randn((num_rows, pad_begin_len)) * self.noise_level
            pad_end = torch.randn((num_rows, pad_end_len)) * self.noise_level

            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return sig