import torch
import torch.nn as nn
####################################################################################################
# This is for models used in revision. Currently only containing 
####################################################################################################
# Our model sizing keeps the linear layer in-outs same, while only changing the convolutional layers
####################################################################################################
# These three are CNN models
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
    
# Large Model: we used this - 20250819
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


class CNNAutoencoder(nn.Module):
    """
    20250815
    Autoencoder version of the large CNN with three CNN blocks.

    - Encoder: three CNN blocks + linear
    - Decoder: linear + three CNN blocks (mirrored)
    - get_representation(x): returns the output of encoder linear
    """
    def __init__(self, input_shape, hidden_dim=256, n_filter_base=4, n_filter_exp=2, dropout_rate=0.5):
        super().__init__()

        n_filter_1 = pow(n_filter_base, n_filter_exp)
        n_filter_2 = pow(n_filter_base, n_filter_exp+1)
        n_filter_3 = pow(n_filter_base, n_filter_exp+2)

        self.last_cnn_channel = n_filter_3
        self.last_cnn_height = input_shape[2] // (pow(2, 3))
        self.last_cnn_width = input_shape[3] // (pow(2, 3))

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, n_filter_1, kernel_size=3, stride=1, padding='same'), 
            nn.BatchNorm2d(n_filter_1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(n_filter_1, n_filter_2, kernel_size=3, stride=1, padding='same'), 
            nn.BatchNorm2d(n_filter_2), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(n_filter_2, n_filter_3, kernel_size=3, stride=1, padding='same'), 
            nn.BatchNorm2d(n_filter_3), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
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
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(n_filter_3, n_filter_2, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(n_filter_2), 
            nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(n_filter_2, n_filter_1, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(n_filter_1), 
            nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(n_filter_1, 1, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(1), 
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

    def forward(self, x, return_latent: bool = False):
        # x_shape = [batch_size, n_channel = 1, height, width]

        # encoding
        x = self.dropout(self.encoder_conv(x)) # [batch_size, n_channel = 4^4, height/2^3, width/2^3]
        x = x.view(x.shape[0], -1) # [batch_size, 4^4 * height/2^3 * width/2^3]
        z = self.encoder_fc(x) # [batch_size, hidden_dim]

        # decoding
        recon = self.dropout(self.decoder_fc(z)) # [batch_size, 4^4 * height/2^3 * width/2^3]
        recon = recon.view(recon.shape[0], self.last_cnn_channel, self.last_cnn_height, self.last_cnn_width) # [batch_size, 4^4, height/2^3, width/2^3]
        recon = self.decoder_conv(recon) #[batch_size, 1, height, width]

        if return_latent:
            return recon, z
        return recon
    
    def get_representation(self, x):
        with torch.no_grad():
            x = self.dropout(self.encoder_conv(x))
            x = x.view(x.shape[0], -1)
            x = self.encoder_fc(x)
        return x


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


class ResLinearAutoencoder(nn.Module):
    """
    20250814
    Autoencoder version of the residual linear network.

    - Encoder: self.res_blocks (same as before)
    - Decoder: self.final_lin (latent -> input_dim) for reconstruction
    - get_representation(x): returns the hidden rep (output of self.res_blocks)
    """
    def __init__(self, input_shape):
        super().__init__()
        in_size = input_shape[2] * input_shape[3]
        hidden_sizes = [in_size, 512, 128]  # keep your original input flattening convention

        # ----- Encoder: same residual stack you had -----
        enc_layers = []
        for i in range(len(hidden_sizes) - 1):
            enc_layers.append(ResidualBlock(hidden_sizes[i], hidden_sizes[i+1]))
        self.encoder = nn.Sequential(*enc_layers)

        # ----- Decoder (mirror) -----
        dec_sizes = hidden_sizes[::-1]          # e.g., [128, 512, in_size]
        dec_layers = []
        for i in range(len(dec_sizes) - 1):
            dec_layers.append(ResidualBlock(dec_sizes[i], dec_sizes[i+1]))
        self.decoder = nn.Sequential(*dec_layers)

        # ----- Inits: same helpers you used -----
        self.encoder.apply(self.init_res_weights)
        self.decoder.apply(self.init_res_weights)

    def init_res_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.1)
            m.bias.data.fill_(0.01)

    def init_lin_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.1)
            m.bias.data.fill_(0.01)

    def forward(self, x, return_latent: bool = False):
        # Remember original (batch-first) shape, e.g. (B, C, D, L)
        orig_shape = x.shape
        # Flatten input exactly like your classifier version
        x_flat = x.view(x.size(0), -1)

        # Encoder (hidden representation)
        z = self.encoder(x_flat)

        # Decoder head to reconstruct the flat input
        recon_flat = self.decoder(z)

        # Reshape back to *exactly* the input shape
        recon = recon_flat.view(orig_shape)

        if return_latent:
            return recon, z
        return recon

    # ---- New: expose the hidden representation cleanly ----
    def get_representation(self, x):
        with torch.no_grad():
            x_flat = x.view(x.size(0), -1)
            z = self.res_blocks(x_flat)
        return z


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


class LSTMAutoencoder(nn.Module):
    """
    20250814
    LSTM Autoencoder version of your original LSTMNetwork.
    - Encoder LSTM + Linear projection to latent_dim
    - Decoder LSTM fed with repeated latent vector
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # ----- Encoder LSTM -----
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )

        # Projection layer: reduce from 2*128 to latent_dim
        self.enc_proj = nn.Linear(2 * 128, latent_dim)

        # ----- Decoder LSTM -----
        self.dec_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            bidirectional=False
        )

        # Output projection back to feature size
        self.linear = nn.Linear(128, 64)

        # Weight initialization
        self.lstm.apply(self.init_lstm_weights)
        self.enc_proj.apply(self.init_lin_weights)
        self.dec_lstm.apply(self.init_lstm_weights)
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

    def forward(self, x, return_latent: bool = False):
        # Remember original shape: (B, 1, D, L)
        orig_shape = x.shape
        B, _, D, L = orig_shape

        # (B, 1, D, L) -> (B, L, D)
        x_seq = x.squeeze(1).transpose(1, 2)

        # Encode
        enc_out, _ = self.lstm(x_seq)              # (B, L, 256)
        last_timestep = enc_out[:, -1, :]          # (B, 256)
        z = self.enc_proj(last_timestep)           # (B, latent_dim)

        # Decode
        L = x_seq.size(1)
        dec_in = z.unsqueeze(1).repeat(1, L, 1)    # (B, L, latent_dim)
        dec_out, _ = self.dec_lstm(dec_in)         # (B, L, 128)
        recon = self.linear(dec_out)               # (B, L, 64)

        # Back to (B, 1, D, L) to match input
        recon = recon.transpose(1, 2).unsqueeze(1)  # (B, 1, D, L)

        if return_latent:
            return recon, z
        return recon

    def get_representation(self, x):
        with torch.no_grad():
            x_seq = x.squeeze(1).transpose(1, 2)
            enc_out, _ = self.lstm(x_seq)
            last_timestep = enc_out[:, -1, :]
            z = self.enc_proj(last_timestep)
        return z