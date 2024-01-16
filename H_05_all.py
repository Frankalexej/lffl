### Consonant and Vowel separated
### Why? In one paper it is reported that preterm babies have attenuated performance only on vowels. 

# All in Runner
## Importing the libraries
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import torchaudio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from torchinfo import summary
import torch.nn.functional as F
from torch.nn import init
from model_model import SelfPackLSTM
from model_configs import ModelDimConfigs, TrainingConfigs
from misc_tools import get_timestamp, ARPABET
from model_dataset import DS_Tools, Padder, TokenMap, NormalizerKeepShape
from model_dataset import SingleRecSelectBalanceDatasetPrecombine as ThisDataset
from model_filter import XpassFilter
from paths import *
from ssd_paths import *
from misc_progress_bar import draw_progress_bar
from misc_recorder import *

def draw_learning_curve_and_accuracy(losses, accs, epoch="", best_val=None, save=False, save_name=""): 
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    train_losses, valid_losses, full_valid_losses, best_val_loss = losses
    train_accs, valid_accs, full_valid_accs = accs

    # Plot Loss on the left subplot
    ax1.plot(train_losses, label='Train')
    ax1.plot(valid_losses, label='Valid')
    ax1.plot(full_valid_losses, label='Full Valid')
    ax1.axvline(x=best_val_loss, color='r', linestyle='--', label=f'Best: {best_val_loss}')
    ax1.set_title("Learning Curve Loss" + f" {epoch}")
    ax1.legend(loc="upper right")

    # Plot Accuracy on the right subplot
    ax2.plot(train_accs, label='Train')
    ax2.plot(valid_accs, label='Valid')
    ax2.plot(full_valid_accs, label='Full Valid')
    if best_val: 
        ax2.axhline(y=best_val, color='r', linestyle='--', label=f'Best: {best_val:.3f}')
    ax2.set_title('Learning Curve Accuracy' + f" {epoch}")
    ax2.legend(loc="lower right")

    # Display the plots
    plt.tight_layout()
    plt.xlabel("Epoch")
    display.clear_output(wait=True)
    display.display(plt.gcf())
    if save: 
        plt.savefig(save_name)

### Prepare Model Definition
class Network(nn.Module):
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
            nn.Linear(128, 64),  # Reduced size
        )
        self.lin = nn.Linear(in_features=64, out_features=38)

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

def load_data(type="f", sel="full"):
    if type == "l":
        mytrans = nn.Sequential(
            Padder(sample_rate=TrainingConfigs.REC_SAMPLE_RATE, pad_len_ms=250, noise_level=1e-4), 
            XpassFilter(cut_off_upper=500),
            torchaudio.transforms.MelSpectrogram(TrainingConfigs.REC_SAMPLE_RATE, 
                                                n_mels=TrainingConfigs.N_MELS, 
                                                n_fft=TrainingConfigs.N_FFT, 
                                                power=2), 
            torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80), 
            NormalizerKeepShape(NormalizerKeepShape.norm_mvn)
        )
    elif type == "h": 
        mytrans = nn.Sequential(
            Padder(sample_rate=TrainingConfigs.REC_SAMPLE_RATE, pad_len_ms=250, noise_level=1e-4), 
            XpassFilter(cut_off_upper=10000, cut_off_lower=4000),
            torchaudio.transforms.MelSpectrogram(TrainingConfigs.REC_SAMPLE_RATE, 
                                                n_mels=TrainingConfigs.N_MELS, 
                                                n_fft=TrainingConfigs.N_FFT, 
                                                power=2), 
            torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80), 
            NormalizerKeepShape(NormalizerKeepShape.norm_mvn)
        )
    else: 
        mytrans = nn.Sequential(
            Padder(sample_rate=TrainingConfigs.REC_SAMPLE_RATE, pad_len_ms=250, noise_level=1e-4), 
            torchaudio.transforms.MelSpectrogram(TrainingConfigs.REC_SAMPLE_RATE, 
                                                n_mels=TrainingConfigs.N_MELS, 
                                                n_fft=TrainingConfigs.N_FFT, 
                                                power=2), 
            torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80), 
            NormalizerKeepShape(NormalizerKeepShape.norm_mvn)
        )
    with open(os.path.join(src_, "no-stress-seg.dict"), "rb") as file:
        # Load the object from the file
        mylist = pickle.load(file)
        mylist.remove('AH') # we don't include this, it is too mixed. 
    if sel == "c": 
        select = ARPABET.intersect_lists(mylist, ARPABET.list_consonants())
    elif sel == "v":
        select = ARPABET.intersect_lists(mylist, ARPABET.list_vowels())
    else:
        select = mylist
    # Now you can use the loaded object
    mymap = TokenMap(mylist)
    train_ds = ThisDataset(strain_cut_audio_, 
                        os.path.join(suse_, "guide_train.csv"), 
                        select=select, 
                        mapper=mymap, 
                        transform=mytrans)
    valid_ds = ThisDataset(strain_cut_audio_, 
                        os.path.join(suse_, "guide_validation.csv"), 
                        select=select, 
                        mapper=mymap,
                        transform=mytrans)
    train_ds_indices = DS_Tools.read_indices(os.path.join(model_save_dir, f"train_{sel}.use"))
    valid_ds_indices = DS_Tools.read_indices(os.path.join(model_save_dir, f"valid_{sel}.use"))
    use_train_ds = torch.utils.data.Subset(train_ds, train_ds_indices)
    use_valid_ds = torch.utils.data.Subset(valid_ds, valid_ds_indices)
    train_loader = DataLoader(use_train_ds, batch_size=TrainingConfigs.BATCH_SIZE, 
                            shuffle=True, 
                            num_workers=TrainingConfigs.LOADER_WORKER)
    valid_loader = DataLoader(use_valid_ds, batch_size=TrainingConfigs.BATCH_SIZE, 
                            shuffle=False, 
                            num_workers=TrainingConfigs.LOADER_WORKER)
    return train_loader, valid_loader

def run_once(hyper_dir, pretype="f", posttype="f", sel="full"): 
    model_save_dir = os.path.join(hyper_dir, sel, f"{pretype}{posttype}")
    mk(model_save_dir)

    # Loss Recording
    train_losses = ListRecorder(os.path.join(model_save_dir, "train.loss"))
    valid_losses = ListRecorder(os.path.join(model_save_dir, "valid.loss"))
    full_valid_losses = ListRecorder(os.path.join(model_save_dir, "full_valid.loss"))
    train_accs = ListRecorder(os.path.join(model_save_dir, "train.acc"))
    valid_accs = ListRecorder(os.path.join(model_save_dir, "valid.acc"))
    full_valid_accs = ListRecorder(os.path.join(model_save_dir, "full_valid.acc"))
    special_recs = DictRecorder(os.path.join(model_save_dir, "special.hst"))

    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    model = Network()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model_str = str(model)
    model_txt_path = os.path.join(model_save_dir, "model.txt")
    with open(model_txt_path, "w") as f:
        f.write(model_str)
        f.write("\n")
        f.write(str(summary(model, input_size=(128, 1, 64, 21))))

    # Load Data (I&II)
    train_loader_1, valid_loader_1 = load_data(type=pretype, sel=sel)
    train_loader_2, valid_loader_2 = load_data(type=posttype, sel=sel)

    # Train (I)
    best_valid_loss = 1e9
    best_valid_loss_epoch = 0
    EPOCHS = 3
    BASE = 0

    for epoch in range(BASE, BASE + EPOCHS):
        model.train()
        train_loss = 0.
        train_num = len(train_loader_1)    # train_loader
        train_correct = 0
        train_total = 0
        for idx, (x, y) in enumerate(train_loader_1):
            optimizer.zero_grad()
            x = x.to(device)
            y = torch.tensor(y, device=device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            pred = model.predict_on_output(y_hat)
            train_total += y_hat.size(0)
            train_correct += (pred == y).sum().item()
            draw_progress_bar(idx, train_num, title="Train")

        train_losses.append(train_loss / train_num)
        train_accs.append(train_correct / train_total)
        last_model_name = f"{epoch}.pt"
        torch.save(model.state_dict(), os.path.join(model_save_dir, last_model_name))

        model.eval()
        valid_loss = 0.
        valid_num = len(valid_loader_1)
        valid_correct = 0
        valid_total = 0
        for idx, (x, y) in enumerate(valid_loader_1):
            x = x.to(device)
            y = torch.tensor(y, device=device)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            valid_loss += loss.item()

            pred = model.predict_on_output(y_hat)

            valid_total += y_hat.size(0)
            valid_correct += (pred == y).sum().item()

        avg_valid_loss = valid_loss / valid_num
        valid_losses.append(avg_valid_loss)
        valid_accs.append(valid_correct / valid_total)
        if avg_valid_loss < best_valid_loss: 
            best_valid_loss = avg_valid_loss
            best_valid_loss_epoch = epoch

        model.eval()
        full_valid_loss = 0.
        full_valid_num = len(valid_loader_2)
        full_valid_correct = 0
        full_valid_total = 0
        for idx, (x, y) in enumerate(valid_loader_2):
            x = x.to(device)
            y = torch.tensor(y, device=device)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            full_valid_loss += loss.item()

            pred = model.predict_on_output(y_hat)

            full_valid_total += y_hat.size(0)
            full_valid_correct += (pred == y).sum().item()
        full_valid_losses.append(full_valid_loss / full_valid_num)
        full_valid_accs.append(full_valid_correct / full_valid_total)

        train_losses.save()
        valid_losses.save()
        full_valid_losses.save()
        train_accs.save()
        valid_accs.save()
        full_valid_accs.save()
        if epoch % 5 == 0:
            draw_learning_curve_and_accuracy(losses=(train_losses.get(), valid_losses.get(), full_valid_losses.get(), best_valid_loss_epoch), 
                                    accs=(train_accs.get(), valid_accs.get(), full_valid_accs.get()), 
                                    epoch=str(epoch), 
                                    save=True, 
                                    save_name=f"{model_save_dir}/vis.png")

    draw_learning_curve_and_accuracy(losses=(train_losses.get(), valid_losses.get(), full_valid_losses.get(), best_valid_loss_epoch), 
                                    accs=(train_accs.get(), valid_accs.get(), full_valid_accs.get()), 
                                    epoch=str(BASE + EPOCHS - 1), 
                                    best_val=valid_accs.get()[best_valid_loss_epoch], 
                                    save=True, 
                                    save_name=f"{model_save_dir}/vis.png")
    
    # Pre Model Best
    special_recs.append(("preval_epoch", best_valid_loss_epoch))
    special_recs.append(("preval_acc", valid_accs.get()[best_valid_loss_epoch]))
    special_recs.save()

    # Train (II)
    BASE = BASE + EPOCHS
    EPOCHS = 2
    for epoch in range(BASE, BASE + EPOCHS):
        model.train()
        train_loss = 0.
        train_num = len(train_loader_2)    # train_loader
        train_correct = 0
        train_total = 0
        for idx, (x, y) in enumerate(train_loader_2):
            optimizer.zero_grad()
            x = x.to(device)
            y = torch.tensor(y, device=device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            pred = model.predict_on_output(y_hat)
            train_total += y_hat.size(0)
            train_correct += (pred == y).sum().item()
            draw_progress_bar(idx, train_num, title="Train")

        train_losses.append(train_loss / train_num)
        train_accs.append(train_correct / train_total)
        last_model_name = f"{epoch}.pt"
        torch.save(model.state_dict(), os.path.join(model_save_dir, last_model_name))

        model.eval()
        valid_loss = 0.
        valid_num = len(valid_loader_2)
        valid_correct = 0
        valid_total = 0
        for idx, (x, y) in enumerate(valid_loader_2):
            x = x.to(device)
            y = torch.tensor(y, device=device)

            y_hat = model(x)
            loss = criterion(y_hat, y)
            valid_loss += loss.item()

            pred = model.predict_on_output(y_hat)

            valid_total += y_hat.size(0)
            valid_correct += (pred == y).sum().item()

        avg_valid_loss = valid_loss / valid_num
        valid_losses.append(avg_valid_loss)
        valid_accs.append(valid_correct / valid_total)
        full_valid_losses.append(avg_valid_loss)
        full_valid_accs.append(valid_correct / valid_total)
        if avg_valid_loss < best_valid_loss: 
            best_valid_loss = avg_valid_loss
            best_valid_loss_epoch = epoch

        train_losses.save()
        valid_losses.save()
        train_accs.save()
        valid_accs.save()
        full_valid_losses.save()
        full_valid_accs.save()

        if epoch % 5 == 0:
            draw_learning_curve_and_accuracy(losses=(train_losses.get(), valid_losses.get(), full_valid_losses.get(), best_valid_loss_epoch), 
                                    accs=(train_accs.get(), valid_accs.get(), full_valid_losses.get()), 
                                    epoch=str(epoch), 
                                    save=True, 
                                    save_name=f"{model_save_dir}/vis.png")

    draw_learning_curve_and_accuracy(losses=(train_losses.get(), valid_losses.get(), full_valid_accs.get(), best_valid_loss_epoch), 
                                    accs=(train_accs.get(), valid_accs.get(), full_valid_accs.get()), 
                                    epoch=str(BASE + EPOCHS - 1), 
                                    best_val=valid_accs.get()[best_valid_loss_epoch], 
                                    save=True, 
                                    save_name=f"{model_save_dir}/vis.png")
    
    # Post Model Best
    special_recs.append(("postval_epoch", best_valid_loss_epoch))
    special_recs.append(("postval_acc", valid_accs.get()[best_valid_loss_epoch]))
    special_recs.save()

if __name__ == "__main__": 
    RUN_TIMES = 1
    for run_time in range(RUN_TIMES):
        ## Hyper-preparations
        ts = str(get_timestamp())
        train_name = "H05"
        model_save_dir = os.path.join(model_save_, f"{train_name}-{ts}")
        print(f"{train_name}-{ts}")
        mk(model_save_dir)

        ### Get Data (Not Loading)
        mytrans = None

        with open(os.path.join(src_, "no-stress-seg.dict"), "rb") as file:
            # Load the object from the file
            mylist = pickle.load(file)
            mylist.remove("AH")

        select_consonants = ARPABET.intersect_lists(mylist, ARPABET.list_consonants())
        select_vowels = ARPABET.intersect_lists(mylist, ARPABET.list_vowels())

        mymap = TokenMap(mylist)
        for select, savename, use_proportion in zip([select_consonants, select_vowels], 
                                                           ["c", "v"], 
                                                           [0.01, 0.02]):
            train_ds = ThisDataset(strain_cut_audio_, 
                                os.path.join(suse_, "guide_train.csv"), 
                                select=select, 
                                mapper=mymap, 
                                transform=mytrans)
            valid_ds = ThisDataset(strain_cut_audio_, 
                                os.path.join(suse_, "guide_validation.csv"), 
                                select=select, 
                                mapper=mymap,
                                transform=mytrans)

            # train data
            use_len = int(use_proportion * len(train_ds))
            remain_len = len(train_ds) - use_len
            use_train_ds, remain_ds = random_split(train_ds, [use_len, remain_len])

            # valid data
            use_len = int(use_proportion * len(valid_ds))
            remain_len = len(valid_ds) - use_len
            use_valid_ds, remain_ds = random_split(valid_ds, [use_len, remain_len])

            # NOTE: we don't need to save the cut-small subset, because after cutting-small, 
            # the saved train and valid separations will reflect this
            DS_Tools.save_indices(os.path.join(model_save_dir, f"train_{savename}.use"), use_train_ds.indices)
            DS_Tools.save_indices(os.path.join(model_save_dir, f"valid_{savename}.use"), use_valid_ds.indices)
            print(len(use_train_ds), len(use_valid_ds))
        
        for sel_type in ["c", "v"]: 
            run_once(model_save_dir, pretype="f", posttype="f", sel=sel_type)
            run_once(model_save_dir, pretype="l", posttype="f", sel=sel_type)
            run_once(model_save_dir, pretype="h", posttype="f", sel=sel_type)
