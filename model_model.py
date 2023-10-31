import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR

from model_configs import ModelDimConfigs

def extract_last_from_packed(packed, lengths:torch.Tensor): 
    sum_batch_sizes = torch.cat((
        torch.zeros(2, dtype=torch.int64),
        torch.cumsum(packed.batch_sizes, 0)
    ))
    sorted_lengths = lengths[packed.sorted_indices]
    last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.size(0))
    last_seq_items = packed.data[last_seq_idxs]
    last_seq_items = last_seq_items[packed.unsorted_indices]
    return last_seq_items

class SelfPackLSTM(nn.Module): 
    """
    This is a packing class that includes pack_padded_sequence 
    and pad_packed_sequence into the RNN class (LSTM)
    The output is the last items of the batch. So (B, L, I) -> (B, L, O) -> (B, O) 
    """
    def __init__(self, in_size, out_size, num_layers=1):
        super(SelfPackLSTM, self).__init__()
        # get resnet model
        self.rnn = nn.LSTM(input_size=in_size, 
                           hidden_size=out_size, 
                           num_layers=num_layers, 
                           batch_first=True)
        
    
    def forward(self, x, x_lens): 
        x = pack_padded_sequence(x, x_lens, 
                                 batch_first=True, 
                                 enforce_sorted=False)
        
        x, (hn, cn) = self.rnn(x)   # (B, L, I) -> (B, L, O)
        # x, _ = pad_packed_sequence(x, batch_first=True)
        x = extract_last_from_packed(x, x_lens) # extract the last elements
        return x


class SelfPackLSTMNetron(nn.Module): 
    """
    This is a packing class that includes pack_padded_sequence 
    and pad_packed_sequence into the RNN class (LSTM)
    The output is the last items of the batch. So (B, L, I) -> (B, L, O) -> (B, O) 
    """
    def __init__(self, in_size, out_size, num_layers=1):
        super(SelfPackLSTMNetron, self).__init__()
        # get resnet model
        self.rnn = nn.LSTM(input_size=in_size, 
                           hidden_size=out_size, 
                           num_layers=num_layers, 
                           batch_first=True)
        
    
    def forward(self, x, x_lens): 
        x = pack_padded_sequence(x, x_lens, 
                                 batch_first=True, 
                                 enforce_sorted=False)
        
        x, (hn, cn) = self.rnn(x)   # (B, L, I) -> (B, L, O)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x[:, -1, :]
        # x = extract_last_from_packed(x, x_lens) # extract the last elements
        return x

class SiameseNetwork(nn.Module):
    """
        Siamese network for phone similarity prediction.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        In addition, we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick. [? might try later]
    """
    def __init__(self, dimconf:ModelDimConfigs, num_layers=2):
        super(SiameseNetwork, self).__init__()
        # get resnet model
        self.rnn = SelfPackLSTM(in_size=dimconf.rnn_in_size, 
                                out_size=dimconf.rnn_out_size, 
                                num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(dimconf.lin_in_size_1 * 2, dimconf.lin_out_size_1),
            nn.ReLU(),
            nn.Linear(dimconf.lin_in_size_2, dimconf.lin_out_size_2),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.rnn.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x, x_lens):
        # get through the rnn
        output = self.rnn(x, x_lens)
        # output = output.view(output.size()[0], -1)
        return output

    def forward(self, inputs, inputs_lens):
        input1, input2 = inputs
        input1_lens, input2_lens = inputs_lens
        # get two images' features
        output1 = self.forward_once(input1, input1_lens)
        output2 = self.forward_once(input2, input2_lens)

        # concatenate both images' features
        # (B, F) -> (B, 2F)
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        
        return output
    

class JudgeNetwork(nn.Module):
    def __init__(self, dimconf:ModelDimConfigs, num_layers=2):
        super(JudgeNetwork, self).__init__()
        self.rnn = SelfPackLSTM(in_size=dimconf.rnn_in_size, 
                                out_size=dimconf.rnn_out_size, 
                                num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(dimconf.lin_in_size_1, dimconf.lin_out_size_1),
            nn.ReLU(),
            nn.Linear(dimconf.lin_in_size_2, dimconf.lin_out_size_2),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.rnn.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x, x_lens):
        output = self.rnn(x, x_lens)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        
        return output
    
