import torch
import torch.nn as nn

####################################################################################################
# added CNN reconstruction model
# only keeping large CNN following model desciption in OSF
####################################################################################################

# Large Model: three CNN blocks
class CNNAutoencoder(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_filter_base, n_filter_exp,
                 kernel_size, pool_size, padding, dropout_rate):
        super().__init__()

        n_filter_1 = pow(n_filter_base, n_filter_exp)
        n_filter_2 = pow(n_filter_base, n_filter_exp+1)
        n_filter_3 = pow(n_filter_base, n_filter_exp+2)

        self.last_cnn_channel = n_filter_3
        self.last_cnn_height = input_shape[2] // (pow(pool_size, 3))
        self.last_cnn_width = input_shape[3] // (pow(pool_size, 3))

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, n_filter_1, kernel_size=kernel_size, stride=1, padding=padding), 
            #nn.BatchNorm2d(n_filter_1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=pool_size, stride=1), 
            nn.Conv2d(n_filter_1, n_filter_2, kernel_size=kernel_size, stride=1, padding=padding), 
            #nn.BatchNorm2d(n_filter_2), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size), 
            nn.Conv2d(n_filter_2, n_filter_3, kernel_size=kernel_size, stride=1, padding=padding), 
            #nn.BatchNorm2d(n_filter_3), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        )
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(in_features=self.last_cnn_channel * self.last_cnn_height * self.last_cnn_width, 
                      out_features=hidden_dim), 
            nn.ReLU()
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(in_features=hidden_dim, 
                      out_features=self.last_cnn_channel * self.last_cnn_height * self.last_cnn_width), 
            nn.ReLU()
        )

        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=pool_size, mode='nearest'),
            nn.Conv2d(n_filter_3, n_filter_2, kernel_size=kernel_size, stride=1, padding=padding),
            #nn.BatchNorm2d(n_filter_2), 
            nn.ReLU(), 
            nn.Upsample(scale_factor=pool_size, mode='nearest'),
            nn.Conv2d(n_filter_2, n_filter_1, kernel_size=kernel_size, stride=1, padding=padding),
            #nn.BatchNorm2d(n_filter_1), 
            nn.ReLU(), 
            nn.Upsample(scale_factor=pool_size, mode='nearest'),
            nn.Conv2d(n_filter_1, 1, kernel_size=kernel_size, stride=1, padding=padding),
            #nn.BatchNorm2d(1), 
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.encoder_conv.apply(self.init_conv_weights)
        self.encoder_fc.apply(self.init_lin_weights)
        self.decoder_conv.apply(self.init_conv_weights)
        self.decoder_fc.apply(self.init_lin_weights)

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
        # x_shape = [batch_size, n_channel = 1, height, width]

        # encoding
        x = self.dropout(self.encoder_conv(x)) # [batch_size, n_channel = 4^4, height/2^3, width/2^3]
        x = x.view(x.shape[0], -1) # [batch_size, 4^4 * height/2^3 * width/2^3]
        x = self.encoder_fc(x) # [batch_size, hidden_dim]

        # decoding
        x = self.dropout(self.decoder_fc(x)) # [batch_size, 4^4 * height/2^3 * width/2^3]
        x = x.view(x.shape[0], self.lastcnn_channel, self.lastcnn_height, self.lastcnn_width) # [batch_size, 4^4, height/2^3, width/2^3]
        x = self.decoder_conv(x) #[batch_size, 1, height, width]

        return x
    
    def get_representation(self, x):
        with torch.no_grad():
            x = self.dropout(self.encoder_conv(x))
            x = x.view(x.shape[0], -1)
            x = self.encoder_fc(x)
        return x
