import torch
import torchaudio
import random
import pandas as pd
import os

from misc_tools import AudioCut

class PairRecDataset():
    def __init__(self, aud_dir, log_file, total_number=10000, transform=None):
        # we define a total number so that we can freely adjust the total number of training examples we take hold of.
        # Because we definitely don't use up the whole pair set.
        self.aud_dir = aud_dir
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
            # audio_2, sample_rate = torchaudio.load(path_2, normalize=True)

            # set the label for this example to be negative (0)
            # target = torch.tensor(0, dtype=torch.float)
            target = 0.

        return path_1, path_2, target

    def get_list(self):

        # a dataframe where each entry is a pair of audio
        self.paired_audio = pd.DataFrame(columns=["path_1", "path_2", "target"])

        # get pairs for total_number of times and append to dataframe
        for i in range(self.total_number):
            new_line = pd.Series(self.get_pair(i), index=self.paired_audio.columns)
            self.paired_audio = self.paired_audio._append(new_line, ignore_index=True)

        # write to csv
        self.paired_audio.to_csv(os.path.join(self.aud_dir, "paired_dataset"), index=False)

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
