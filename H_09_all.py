### Consonant and Vowel separated
### Why? In one paper it is reported that preterm babies have attenuated performance only on vowels. 
### In H_09, we will train using both consonants and vowels, but for testing only either of them. 
### Also, we want to unify the running of models. We will name the models with names Small, Medium, Large, and place them in models. 

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
from H_10_models import SmallNetwork, MediumNetwork, LargeNetwork
from model_configs import ModelDimConfigs, TrainingConfigs
from misc_tools import get_timestamp, ARPABET
from model_dataset import DS_Tools, Padder, TokenMap, NormalizerKeepShape
from model_dataset import SingleRecSelectBalanceDatasetPrecombine as ThisDataset
from model_filter import XpassFilter
from paths import *
from ssd_paths import *
from misc_progress_bar import draw_progress_bar
from misc_recorder import *
from H_11_drawer import draw_learning_curve_and_accuracy

# Data Loader
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

def run_once(hyper_dir, model_type="large", pretype="f", posttype="f", sel="full"): 
    model_save_dir = os.path.join(hyper_dir, model_type, sel, f"{pretype}{posttype}")
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
    if model_type == "small":
        model = SmallNetwork()
    elif model_type == "medium":
        model = MediumNetwork()
    else:
        model = LargeNetwork()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model_str = str(model)
    model_txt_path = os.path.join(model_save_dir, "model.txt")
    with open(model_txt_path, "w") as f:
        f.write(model_str)
        f.write("\n")
        f.write(str(summary(model, input_size=(128, 1, 64, 21))))

    # Load Data (I&II)
    _, _ = load_data(type=pretype, sel=sel)
    train_loader_1, valid_loader_1 = load_data(type=pretype, sel="full")
    _, _ = load_data(type=posttype, sel=sel)
    train_loader_2, valid_loader_2 = load_data(type=posttype, sel="full")
    # In this way, we get training data will both consonants and vowels, but validation data with only either consonants or vowels. 
    # But the sound range always follows the pretype and posttype settings. 

    # Train (I)
    best_valid_loss = 1e9
    best_valid_loss_epoch = 0
    EPOCHS = 20
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

        # Target Eval
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

        # Full Eval
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
    EPOCHS = 20
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
    
    # Post Model Best
    special_recs.append(("postval_epoch", best_valid_loss_epoch))
    special_recs.append(("postval_acc", valid_accs.get()[best_valid_loss_epoch]))
    special_recs.save()

if __name__ == "__main__": 
    RUN_TIMES = 1
    for run_time in range(RUN_TIMES):
        ## Hyper-preparations
        ts = str(get_timestamp())
        train_name = "H09"
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
        select_full = mylist

        mymap = TokenMap(mylist)
        for select, savename, use_proportion in zip([select_consonants, select_vowels, select_full], 
                                                           ["c", "v", "full"], 
                                                           [0.01, 0.02, 0.01]):
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
        
        for model_type in ["large"]: # "small", "medium", 
            for sel_type in ["c", "v"]: 
                run_once(model_save_dir, model_type=model_type, pretype="f", posttype="f", sel=sel_type)
                run_once(model_save_dir, model_type=model_type, pretype="l", posttype="f", sel=sel_type)
                run_once(model_save_dir, model_type=model_type, pretype="h", posttype="f", sel=sel_type)
