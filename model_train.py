# Import Libs
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import graphviz

# Self-defined libs
from model_model import SiameseNetwork
from model_configs import ModelDimConfigs, TrainingConfigs
from misc_tools import get_timestamp

# graphviz.set_jupyter_format('png')


# define 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

dimconf = ModelDimConfigs(
        rnn_in_size=64, 
        lin_in_size_1=32, 
        lin_in_size_2=16, 
        lin_out_size_2=1
    )

model = SiameseNetwork(
    dimconf=dimconf
)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


ts = str(get_timestamp())