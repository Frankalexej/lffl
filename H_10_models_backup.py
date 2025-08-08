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


# Medium Model
class MediumNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(8), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
            # Removed the third convolutional layer
        )
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin_1 = nn.Sequential(
            nn.Linear(32 * 16 * 5, 128),  # Reduced size
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