import torch
import torch.nn as nn

####################################################################################################
# Our model sizing keeps the linear layer in-outs same, while only changing the convolutional layers
####################################################################################################
# Small Model
class SmallNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(16), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
        )
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin_1 = nn.Sequential(
            nn.Linear(16 * 32 * 10, 128),  # Reduced size
            nn.Dropout(0.5),  # Adjusted dropout rate
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Linear(128, 64),  # Reduced size
        )
        self.lin = nn.Linear(in_features=128, out_features=38)

        self.conv.apply(self.init_conv_weights)
        self.lin.apply(self.init_lin_weights)

    def init_lin_weights(self, m):
        if isinstance(m, nn.Linear):
            # torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.kaiming_normal_(m.weight, a=0.1)
            m.bias.data.fill_(0.01)
    
    def init_conv_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, a=0.1)
            m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        # x = self.ap(x)
        # x = x.view(x.shape[0], -1)
        x = x.view(x.shape[0], -1)
        x = self.lin_1(x)
        x = self.lin(x)
        return x

    def predict_on_output(self, output): 
        output = nn.Softmax(dim=1)(output)
        preds = torch.argmax(output, dim=1)
        return preds


class MediumNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(16), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(16, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin_1 = nn.Sequential(
            nn.Linear(256 * 16 * 5, 128),  # Reduced size
            nn.Dropout(0.5),  # Adjusted dropout rate
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.lin = nn.Linear(in_features=128, out_features=38)

        self.conv.apply(self.init_conv_weights)
        self.lin.apply(self.init_lin_weights)

    def init_lin_weights(self, m):
        if isinstance(m, nn.Linear):
            # torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.kaiming_normal_(m.weight, a=0.1)
            m.bias.data.fill_(0.01)
    
    def init_conv_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, a=0.1)
            m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        # x = self.ap(x)
        # x = x.view(x.shape[0], -1)
        x = x.view(x.shape[0], -1)
        x = self.lin_1(x)
        x = self.lin(x)
        return x

    def predict_on_output(self, output): 
        output = nn.Softmax(dim=1)(output)
        preds = torch.argmax(output, dim=1)
        return preds
    
# Large Model
class LargeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(16), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin_1 = nn.Sequential(
            nn.Linear(256 * 8 * 2, 128), 
            nn.Dropout(0.5), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Linear(512, 256),
        )
        self.lin = nn.Linear(in_features=128, out_features=38)

        self.conv.apply(self.init_conv_weights)
        self.lin.apply(self.init_lin_weights)

    def init_lin_weights(self, m):
        if isinstance(m, nn.Linear):
            # torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.kaiming_normal_(m.weight, a=0.1)
            m.bias.data.fill_(0.01)
    
    def init_conv_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, a=0.1)
            m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        # x = self.ap(x)
        # x = x.view(x.shape[0], -1)
        x = x.view(x.shape[0], -1)
        x = self.lin_1(x)
        x = self.lin(x)
        return x

    def predict_on_output(self, output): 
        output = nn.Softmax(dim=1)(output)
        preds = torch.argmax(output, dim=1)
        return preds


class ResidualBlock(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.lin1 = nn.Linear(input_features, output_features)
        self.bn1 = nn.BatchNorm1d(output_features)
        self.relu = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(output_features, output_features)
        self.bn2 = nn.BatchNorm1d(output_features)
        
        # If input and output features are the same, we can use a direct identity shortcut
        # Otherwise, we should have a linear transformation for the shortcut
        self.shortcut = nn.Sequential()
        if input_features != output_features:
            self.shortcut = nn.Sequential(
                nn.Linear(input_features, output_features),
                nn.BatchNorm1d(output_features)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.lin1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.lin2(out)
        out = self.bn2(out)
        
        out += identity  # Element-wise addition
        out = self.relu(out)
        
        return out

class ResLinearNetwork(nn.Module):
    """
    Note that we will use the same dataset as that for CNN. 
    However, CNN dataset was deliberately set so that the shape is 
    (B, 1, D, L), instead of the usual (B, L, D). So we have to change this 
    by ourselves in the model. 

    But luckily for linear network we don't have this issue. Just punch to flat. 
    """
    def __init__(self):
        super().__init__()
        in_size = 64 * 21
        hidden_sizes = [in_size, 512, 128]  # Example sizes of hidden layers
        
        # Create Residual Blocks
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(ResidualBlock(hidden_sizes[i], hidden_sizes[i+1]))
        
        self.res_blocks = nn.Sequential(*layers)
        self.final_lin = nn.Linear(hidden_sizes[-1], 38)  # Output size = number of classes

        self.res_blocks.apply(self.init_res_weights)
        self.final_lin.apply(self.init_lin_weights)
    
    def init_res_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.1)
            m.bias.data.fill_(0.01)
    
    def init_lin_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.1)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.res_blocks(x)
        x = self.final_lin(x)
        return x
    
    def predict_on_output(self, output): 
        output = nn.Softmax(dim=1)(output)
        preds = torch.argmax(output, dim=1)
        return preds


class LSTMNetwork(nn.Module):
    """
    Note that we will use the same dataset as that for CNN. 
    However, CNN dataset was deliberately set so that the shape is 
    (B, 1, D, L), instead of the usual (B, L, D). So we have to change this 
    by ourselves in the model. 

    For this, we need to change the dimension order of the input. 
    """
    def __init__(self):
        super().__init__()
        # Define the LSTM layer
        self.lstm = nn.LSTM(64, 128, 3, batch_first=True, bidirectional=True)
        # Define the output layer
        self.linear = nn.Linear(2 * 128, 38)
        
        # Initialize weights
        self.lstm.apply(self.init_lstm_weights)
        self.linear.apply(self.init_lin_weights)

    def init_lstm_weights(self, m):
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.01)
    
    def init_lin_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.1)
            m.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = x.squeeze(1).transpose(1, 2) # (B, 1, D, L) -> (B, L, D)
        lstm_out, (hn, cn) = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        x = self.linear(last_time_step_out)
        return x
    
    def predict_on_output(self, output): 
        output = nn.Softmax(dim=1)(output)
        preds = torch.argmax(output, dim=1)
        return preds
