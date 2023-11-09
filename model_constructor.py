import torch
import torchaudio
import random
import pandas as pd
import os

class constructor():
    def __init__(self, log_file, aud_dir):

        self.log = pd.read_csv(log_file)
        self.aud_dir = aud_dir

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

        self.grouped_examples = {}

        for i, row in self.log.iterrows():
            label = row['segment']
            if label not in self.grouped_examples:
                self.grouped_examples[label] = []
            self.grouped_examples[label].append(i)

    def __len__(self):
        return len(self.log)

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
        folders_1 = self.log.loc[selected_index_1, 'file'].split("-")
        id_1 = self.log.loc[selected_index_1, 'id']
        file_1 = str(folders_1[0]) + "-" + str(folders_1[1]) + "-" + str(folders_1[2]) + "-" + str(id_1) + ".flac"
        path_1 = os.path.join(self.aud_dir, folders_1[0], folders_1[1], file_1)
        audio_1 = torchaudio.load(path_1)

        # same class
        if index % 2 == 0:
            # pick a random index for the second sample
            selected_index_2 = random.choice(self.grouped_examples[selected_label])

            # get the second sample
            folders_2 = self.log.loc[selected_index_2, 'file'].split("-")
            id_2 = self.log.loc[selected_index_2, 'id']
            file_2 = str(folders_2[0]) + "-" + str(folders_2[1]) + "-" + str(folders_2[2]) + "-" + str(id_2) + ".flac"
            path_2 = os.path.join(self.aud_dir, folders_2[0], folders_2[1], file_2)
            audio_2 = torchaudio.load(path_2)

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)

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
            folders_2 = self.log.loc[selected_index_2, 'file'].split("-")
            id_2 = self.log.loc[selected_index_2, 'id']
            file_2 = str(folders_2[0]) + "-" + str(folders_2[1]) + "-" + str(folders_2[2]) + "-" + str(id_2) + ".flac"
            path_2 = os.path.join(self.aud_dir, folders_2[0], folders_2[1], file_2)
            audio_2 = torchaudio.load(path_2)

            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.float)

        return audio_1, audio_2, target
