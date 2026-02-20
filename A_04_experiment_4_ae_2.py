"""
H20: 
This runner will try to run this on multiple models: CNN(current), RNN and Linear. 
Since the running logic is all the same, and the only difference lies in the model structure, 
we mainly change the model, while keeping the in and outs all the same. 
H21: 
This is staged running. We want to test the effect of number of "pretraining" epochs. 
"""

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
from A_00_models import CNNAutoencoder, ResLinearAutoencoder, LSTMAutoencoder
from model_configs import ModelDimConfigs, TrainingConfigs
from misc_tools import get_timestamp, ARPABET
from model_dataset import DS_Tools, Padder, TokenMap, NormalizerKeepShapeManual, NormalizerKeepShape
from model_dataset import SingleRecSelectBalanceDatasetPrecombine as ThisDataset
from model_filter import XpassFilter
from paths import *
from ssd_paths import *
from misc_progress_bar import draw_progress_bar
from misc_recorder import *
from H_11_drawer import draw_learning_curve_and_accuracy
import argparse


# Data Loader
def load_data(type="f", sel="full", load="train"):
    # Load MV_config
    with open(os.path.join(src_, "mv_config_20.pkl"), "rb") as file: 
        mv_config = pickle.load(file)

    normalize_mean, normalize_std = mv_config["mean"], mv_config["std"]

    if type == "l":
        mytrans = nn.Sequential(
            Padder(sample_rate=TrainingConfigs.REC_SAMPLE_RATE, pad_len_ms=250, noise_level=1e-4), 
            XpassFilter(cut_off_upper=500),
            torchaudio.transforms.MelSpectrogram(TrainingConfigs.REC_SAMPLE_RATE, 
                                                n_mels=TrainingConfigs.N_MELS, 
                                                n_fft=TrainingConfigs.N_FFT, 
                                                hop_length=TrainingConfigs.HOP_LENGTH, 
                                                power=2), 
            torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80), 
            NormalizerKeepShapeManual(mean=normalize_mean, std=normalize_std)
        )
    elif type == "h": 
        mytrans = nn.Sequential(
            Padder(sample_rate=TrainingConfigs.REC_SAMPLE_RATE, pad_len_ms=250, noise_level=1e-4), 
            XpassFilter(cut_off_upper=10000, cut_off_lower=4000),
            torchaudio.transforms.MelSpectrogram(TrainingConfigs.REC_SAMPLE_RATE, 
                                                n_mels=TrainingConfigs.N_MELS, 
                                                n_fft=TrainingConfigs.N_FFT, 
                                                hop_length=TrainingConfigs.HOP_LENGTH, 
                                                power=2), 
            torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80), 
            NormalizerKeepShapeManual(mean=normalize_mean, std=normalize_std)
        )
    else: 
        mytrans = nn.Sequential(
            Padder(sample_rate=TrainingConfigs.REC_SAMPLE_RATE, pad_len_ms=250, noise_level=1e-4), 
            torchaudio.transforms.MelSpectrogram(TrainingConfigs.REC_SAMPLE_RATE, 
                                                n_mels=TrainingConfigs.N_MELS, 
                                                n_fft=TrainingConfigs.N_FFT, 
                                                hop_length=TrainingConfigs.HOP_LENGTH, 
                                                power=2), 
            torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80), 
            NormalizerKeepShapeManual(mean=normalize_mean, std=normalize_std)
        )
    # if type == "l":
    #     mytrans = nn.Sequential(
    #         Padder(sample_rate=TrainingConfigs.REC_SAMPLE_RATE, pad_len_ms=250, noise_level=1e-4), 
    #         XpassFilter(cut_off_upper=500),
    #         torchaudio.transforms.MelSpectrogram(TrainingConfigs.REC_SAMPLE_RATE, 
    #                                             n_mels=TrainingConfigs.N_MELS, 
    #                                             n_fft=TrainingConfigs.N_FFT, 
    #                                             hop_length=TrainingConfigs.HOP_LENGTH,
    #                                             power=2), 
    #         torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80), 
    #         NormalizerKeepShape(NormalizerKeepShape.norm_mvn)
    #     )
    # elif type == "h": 
    #     mytrans = nn.Sequential(
    #         Padder(sample_rate=TrainingConfigs.REC_SAMPLE_RATE, pad_len_ms=250, noise_level=1e-4), 
    #         XpassFilter(cut_off_upper=10000, cut_off_lower=4000),
    #         torchaudio.transforms.MelSpectrogram(TrainingConfigs.REC_SAMPLE_RATE, 
    #                                             n_mels=TrainingConfigs.N_MELS, 
    #                                             n_fft=TrainingConfigs.N_FFT, 
    #                                             hop_length=TrainingConfigs.HOP_LENGTH,
    #                                             power=2), 
    #         torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80), 
    #         NormalizerKeepShape(NormalizerKeepShape.norm_mvn)
    #     )
    # else: 
    #     mytrans = nn.Sequential(
    #         Padder(sample_rate=TrainingConfigs.REC_SAMPLE_RATE, pad_len_ms=250, noise_level=1e-4), 
    #         torchaudio.transforms.MelSpectrogram(TrainingConfigs.REC_SAMPLE_RATE, 
    #                                             n_mels=TrainingConfigs.N_MELS, 
    #                                             n_fft=TrainingConfigs.N_FFT, 
    #                                             hop_length=TrainingConfigs.HOP_LENGTH,
    #                                             power=2), 
    #         torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80), 
    #         NormalizerKeepShape(NormalizerKeepShape.norm_mvn)
    #     )
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
    if load == "train": 
        train_ds = ThisDataset(strain_cut_audio_, 
                            os.path.join(suse_, "guide_train.csv"), 
                            select=select, 
                            mapper=mymap, 
                            transform=mytrans)
        
        train_ds_indices = DS_Tools.read_indices(os.path.join(model_save_dir, f"train_{sel}.use"))
        use_train_ds = torch.utils.data.Subset(train_ds, train_ds_indices)
        train_loader = DataLoader(use_train_ds, batch_size=TrainingConfigs.BATCH_SIZE, 
                                shuffle=True, 
                                num_workers=TrainingConfigs.LOADER_WORKER)
        
        return train_loader
    elif load == "valid":
        valid_ds = ThisDataset(strain_cut_audio_, 
                            os.path.join(suse_, "guide_validation.csv"), 
                            select=select, 
                            mapper=mymap,
                            transform=mytrans)
        valid_ds_indices = DS_Tools.read_indices(os.path.join(model_save_dir, f"valid_{sel}.use"))
        use_valid_ds = torch.utils.data.Subset(valid_ds, valid_ds_indices)
        valid_loader = DataLoader(use_valid_ds, batch_size=TrainingConfigs.BATCH_SIZE, 
                                shuffle=False, 
                                num_workers=TrainingConfigs.LOADER_WORKER)
        return valid_loader

def draw_learning_curve_and_accuracy(losses, accs, epoch="", best_val=None, save=False, save_name=""): 
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    train_losses, valid_losses, full_valid_losses = losses
    train_accs, valid_accs, full_valid_accs = accs

    # Plot Loss on the left subplot
    ax1.plot(train_losses, label='Train')
    ax1.plot(valid_losses, label='Valid')
    ax1.plot(full_valid_losses, label='Full Valid')
    ax1.set_title("Learning Curve Loss" + f" {epoch}")
    ax1.legend(loc="upper right")

    # Plot Accuracy on the right subplot
    ax2.plot(train_accs, label='Train')
    ax2.plot(valid_accs, label='Valid')
    ax2.plot(full_valid_accs, label='Full Valid')
    ax2.set_title('Learning Curve Accuracy' + f" {epoch}")
    ax2.legend(loc="lower right")

    # Display the plots
    plt.tight_layout()
    plt.xlabel("Epoch")
    display.clear_output(wait=True)
    display.display(plt.gcf())
    if save: 
        plt.savefig(save_name)

def kmeans_evaluate(X, Y, n_clusters=None, n_init=20, random_state=0, compute_sil=True):
    if n_clusters is None:
        n_clusters = len(np.unique(Y))
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    cluster_ids = km.fit_predict(X)
    acc = clustering_accuracy(Y, cluster_ids)
    sil = silhouette_score(X, cluster_ids) if compute_sil and len(np.unique(cluster_ids)) > 1 else None
    return {
        "kmeans_acc": acc,
        "silhouette": sil,
        "cluster_counts": np.bincount(cluster_ids, minlength=n_clusters)
    }
def concat_func(z, y): 
    return np.concatenate(z, axis=0), np.concatenate(y, axis=0)

class EvalSaver: 
    def __init__(self, model_save_dir): 
        self.model_save_dir = model_save_dir
        
    def save_eval_func(self, z, y, name, epoch, save_eval=True): 
        if save_eval: 
            np.save(os.path.join(self.model_save_dir, f"{epoch:04d}_{name}_z.npy"), z)
            np.save(os.path.join(self.model_save_dir, f"{epoch:04d}_{name}_y.npy"), y)

def special_on_site_eval_func(z, y, rec, name, on_site_eval): 
    if on_site_eval: 
        eval_res = kmeans_evaluate(z, y)
        acc = eval_res["kmeans_acc"]
        rec.append(("notrain-target-acc", acc))

def on_site_eval_func(z, y, rec, on_site_eval): 
    if on_site_eval: 
        eval_res = kmeans_evaluate(z, y)
        acc = eval_res["kmeans_acc"]
        rec.append(acc)

def on_site_eval_func_multi(z, y, recs, on_site_eval): 
    if on_site_eval: 
        eval_res = kmeans_evaluate(z, y)
        acc = eval_res["kmeans_acc"]
        for rec in recs: 
            rec.append(acc)

def run_once(hyper_dir, model_type="large", pretype="f", posttype="f", sel="full", preepochs=20, postepochs=20, on_site_eval=False, save_eval=True, eval_train=False, save_model=False): 
    model_save_dir = os.path.join(hyper_dir, f"{model_type}-{preepochs}-{postepochs}", sel, f"{pretype}{posttype}")
    mk(model_save_dir)

    # Loss Recording
    train_losses = ListRecorder(os.path.join(model_save_dir, "train.loss"))
    valid_losses = ListRecorder(os.path.join(model_save_dir, "valid.loss"))
    full_valid_losses = ListRecorder(os.path.join(model_save_dir, "full_valid.loss"))
    trainlikevalid_losses = ListRecorder(os.path.join(model_save_dir, "trainlikevalid.loss"))

    train_accs = ListRecorder(os.path.join(model_save_dir, "train.acc"))
    valid_accs = ListRecorder(os.path.join(model_save_dir, "valid.acc"))
    full_valid_accs = ListRecorder(os.path.join(model_save_dir, "full_valid.acc"))
    trainlikevalid_accs = ListRecorder(os.path.join(model_save_dir, "trainlikevalid.acc"))

    special_recs = DictRecorder(os.path.join(model_save_dir, "special.hst"))

    eval_saver = EvalSaver(model_save_dir)

    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    input_shape = (128, 1, 64, 32)
    if model_type == "cnn": 
        model = CNNAutoencoder(input_shape=input_shape)
    elif model_type == "reslin": 
        model = ResLinearAutoencoder(input_shape=input_shape)
    elif model_type == "lstm": 
        model = LSTMAutoencoder()
    else:
        raise Exception("Model not defined! ")
    # model= nn.DataParallel(model)
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    # NOTE: 20240819 changed lr from 1e-3 to 1e-5 so as to observe learning differences (potentially)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model_str = str(model)
    model_txt_path = os.path.join(model_save_dir, "model.txt")
    with open(model_txt_path, "w") as f:
        f.write(model_str)
        f.write("\n")
        f.write(str(summary(model, input_size=input_shape)))

    # Load Data (I&II)
    train_loader_1 = load_data(type=pretype, sel="full", load="train")
    valid_loader_1 = load_data(type=pretype, sel=sel, load="valid") # target 
    train_loader_2 = load_data(type=posttype, sel="full", load="train")
    valid_loader_2 = load_data(type=posttype, sel=sel, load="valid")    # full = trainlike (because this time we don't separate c/v)
    # trainlikevalid_loader_1 = load_data(type=pretype, sel="full", load="valid")
    # trainlikevalid_loader_2 = load_data(type=posttype, sel="full", load="valid")
    # In this way, we get training data will both consonants and vowels, but validation data with only either consonants or vowels. 
    # But the sound range always follows the pretype and posttype settings. 

    # this is mainly to get the "improvement" for 
    # only-full training models, because they naturally
    # don't have a "transition" from nothing to 
    # "having been trained on full"
    """No Learning Baseline Get"""
    # Target Eval
    model.eval()
    valid_loss = 0.
    valid_num = len(valid_loader_1)
    z_list, y_list = [], []
    for idx, (x, y) in enumerate(valid_loader_1):
        # NOTE: still, x is data, y is label. But instead we will output x_hat, not y_hat. 
        x = x.to(device)
        y = y.to(device)

        x_hat, z = model(x, return_latent=True)
        loss = criterion(x_hat, x) # NOTE: now compare with data (x), not label (y). 
        valid_loss += loss.item()

        z_list.append(z.detach().cpu().numpy())
        y_list.append(y.detach().cpu().numpy())

    special_recs.append(("notrain-target-loss", valid_loss / valid_num))
    z_all, y_all = concat_func(z_list, y_list)
    eval_saver.save_eval_func(z_all, y_all, "valid", 9999, save_eval)
    special_on_site_eval_func(z_all, y_all, special_recs, "notrain-target-acc", on_site_eval)
    special_recs.save()

    # Full Eval
    model.eval()
    full_valid_loss = 0.0
    full_valid_num = len(valid_loader_2)
    z_list, y_list = [], []
    for idx, (x, y) in enumerate(valid_loader_2):
        x = x.to(device)
        y = y.to(device)

        x_hat, z = model(x, return_latent=True)
        loss = criterion(x_hat, x)
        full_valid_loss += loss.item()
        
        z_list.append(z.detach().cpu().numpy())
        y_list.append(y.detach().cpu().numpy())

    special_recs.append(("notrain-full-loss", full_valid_loss / full_valid_num))
    z_all, y_all = concat_func(z_list, y_list)
    eval_saver.save_eval_func(z_all, y_all, "full_valid", 9999, save_eval)
    special_on_site_eval_func(z_all, y_all, special_recs, "notrain-full-acc", on_site_eval)
    special_recs.save()

    # Train (I)
    best_valid_loss = 1e9
    best_valid_loss_epoch = 0
    BASE = 0

    for epoch in range(BASE, BASE + preepochs):
        model.train()
        train_loss = 0.
        train_num = len(train_loader_1)    # train_loader
        z_list, y_list = [], []
        for idx, (x, y) in enumerate(train_loader_1):
            optimizer.zero_grad()
            x = x.to(device)
            # y = torch.tensor(y, device=device)
            y = y.to(device)

            x_hat, z = model(x, return_latent=True)
            loss = criterion(x_hat, x)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            if eval_train: 
                z_list.append(z.detach().cpu().numpy())
                y_list.append(y.detach().cpu().numpy())

        train_losses.append(train_loss / train_num)
        if eval_train: 
            z_all, y_all = concat_func(z_list, y_list)
            eval_saver.save_eval_func(z_all, y_all, "train", epoch, save_eval)
            on_site_eval_func(z_all, y_all, train_accs, on_site_eval)
        if save_model: 
            last_model_name = f"{epoch}.pt"
            torch.save(model.state_dict(), os.path.join(model_save_dir, last_model_name))

        # Target Eval
        model.eval()
        valid_loss = 0.
        valid_num = len(valid_loader_1)
        z_list, y_list = [], []
        for idx, (x, y) in enumerate(valid_loader_1):
            x = x.to(device)
            y = y.to(device)

            x_hat, z = model(x, return_latent=True)
            loss = criterion(x_hat, x)
            valid_loss += loss.item()
            z_list.append(z.detach().cpu().numpy())
            y_list.append(y.detach().cpu().numpy())

        avg_valid_loss = valid_loss / valid_num
        valid_losses.append(avg_valid_loss)
        z_all, y_all = concat_func(z_list, y_list)
        eval_saver.save_eval_func(z_all, y_all, "valid", epoch, save_eval)
        on_site_eval_func(z_all, y_all, valid_accs, on_site_eval)
        
        if avg_valid_loss < best_valid_loss: 
            best_valid_loss = avg_valid_loss
            best_valid_loss_epoch = epoch

        # Full Eval
        model.eval()
        full_valid_loss = 0.
        full_valid_num = len(valid_loader_2)
        z_list, y_list = [], []
        for idx, (x, y) in enumerate(valid_loader_2):
            x = x.to(device)
            y = y.to(device)

            x_hat, z = model(x, return_latent=True)
            loss = criterion(x_hat, x)
            full_valid_loss += loss.item()
            z_list.append(z.detach().cpu().numpy())
            y_list.append(y.detach().cpu().numpy())

        full_valid_losses.append(full_valid_loss / full_valid_num)
        z_all, y_all = concat_func(z_list, y_list)
        eval_saver.save_eval_func(z_all, y_all, "full_valid", epoch, save_eval)
        on_site_eval_func(z_all, y_all, full_valid_accs, on_site_eval)

        train_losses.save()
        valid_losses.save()
        full_valid_losses.save()
        train_accs.save()
        valid_accs.save()
        full_valid_accs.save()

        if epoch % 10 == 0:
            draw_learning_curve_and_accuracy(losses=(train_losses.get(), valid_losses.get(), full_valid_losses.get()), 
                                    accs=(train_accs.get(), valid_accs.get(), full_valid_accs.get()),
                                    epoch=str(epoch), 
                                    save=True, 
                                    save_name=f"{model_save_dir}/vis.png")

    # draw_learning_curve_and_accuracy(losses=(train_losses.get(), valid_losses.get(), full_valid_losses.get()), 
    #                                 accs=(train_accs.get(), valid_accs.get(), full_valid_accs.get()),
    #                                 epoch=str(BASE + preepochs - 1), 
    #                                 save=True, 
    #                                 save_name=f"{model_save_dir}/vis.png")
    
    # Pre Model Best
    special_recs.append(("preval_epoch", best_valid_loss_epoch))
    special_recs.save()

    # Train (II)
    BASE = BASE + preepochs
    for epoch in range(BASE, BASE + postepochs):
        model.train()
        train_loss = 0.
        train_num = len(train_loader_2)    # train_loader
        z_list, y_list = [], []
        for idx, (x, y) in enumerate(train_loader_2):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)

            x_hat, z = model(x, return_latent=True)
            loss = criterion(x_hat, x)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            if eval_train: 
                z_list.append(z.detach().cpu().numpy())
                y_list.append(y.detach().cpu().numpy())

        train_losses.append(train_loss / train_num)
        if eval_train: 
            z_all, y_all = concat_func(z_list, y_list)
            eval_saver.save_eval_func(z_all, y_all, "train", epoch, save_eval)
            on_site_eval_func(z_all, y_all, train_accs, on_site_eval)
        if save_model: 
            last_model_name = f"{epoch}.pt"
            torch.save(model.state_dict(), os.path.join(model_save_dir, last_model_name))

        # Target Eval
        model.eval()
        valid_loss = 0.
        valid_num = len(valid_loader_2)
        z_list, y_list = [], []
        for idx, (x, y) in enumerate(valid_loader_2):
            x = x.to(device)
            y = y.to(device)

            x_hat, z = model(x, return_latent=True)
            loss = criterion(x_hat, x)
            valid_loss += loss.item()
            z_list.append(z.detach().cpu().numpy())
            y_list.append(y.detach().cpu().numpy())


        avg_valid_loss = valid_loss / valid_num
        valid_losses.append(avg_valid_loss)
        full_valid_losses.append(avg_valid_loss)
        z_all, y_all = concat_func(z_list, y_list)
        eval_saver.save_eval_func(z_all, y_all, "valid", epoch, save_eval) # to save storage, we will not save the same data twice. 
        on_site_eval_func_multi(z_all, y_all, [valid_accs, full_valid_accs], on_site_eval)
        
        if avg_valid_loss < best_valid_loss: 
            best_valid_loss = avg_valid_loss
            best_valid_loss_epoch = epoch

        train_losses.save()
        valid_losses.save()
        full_valid_losses.save()
        train_accs.save()
        valid_accs.save()
        full_valid_accs.save()

        if epoch % 10 == 0:
            draw_learning_curve_and_accuracy(losses=(train_losses.get(), valid_losses.get(), full_valid_losses.get()), 
                                    accs=(train_accs.get(), valid_accs.get(), full_valid_accs.get()),
                                    epoch=str(epoch), 
                                    save=True, 
                                    save_name=f"{model_save_dir}/vis.png")

    draw_learning_curve_and_accuracy(losses=(train_losses.get(), valid_losses.get(), full_valid_losses.get()), 
                                    accs=(train_accs.get(), valid_accs.get(), full_valid_accs.get()),
                                    epoch=str(BASE + postepochs - 1), 
                                    save=True, 
                                    save_name=f"{model_save_dir}/vis.png")
    
    # Post Model Best
    special_recs.append(("postval_epoch", best_valid_loss_epoch))
    special_recs.save()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--dataprepare', '-dp', action="store_true")
    parser.add_argument('--timestamp', '-ts', type=str, default="0000000000", help="Timestamp for project, better be generated by bash")
    parser.add_argument('--gpu', '-gpu', type=int, default=0, help="Choose the GPU to work on")
    parser.add_argument('--model','-m',type=str, default = "large",help="Model type: small, medium, large, and others")
    parser.add_argument('--pretype','-p',type=str, default="f", help='Pretraining data type')
    parser.add_argument('--select','-s',type=str, default="full", help='Select full, consonants or vowels')
    parser.add_argument('--preepochs','-pree',type=int, default=20, help='Number of epochs in pre-training')
    parser.add_argument('--postepochs','-poste',type=int, default=20, help='Number of epochs in post-training')
    parser.add_argument('--runnumber','-rn',type=int, default=0, help='The run number')
    

    args = parser.parse_args()
    RUN_TIMES = 1
    for run_time in range(RUN_TIMES):
        ## Hyper-preparations
        # ts = str(get_timestamp())
        ts = args.timestamp
        train_name = "H21"
        model_save_dir = os.path.join(model_save_, f"{train_name}-{ts}")
        print(f"{train_name}-{ts}")
        mk(model_save_dir) 

        if args.dataprepare: 
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
            with open(os.path.join(model_save_dir, f"README.remarks"), "w") as remarks: 
                remarks.write("Normal epoch; lr=1e-3; LargeNetwork, Reslin, LSTM; 012345 10 15 20 25 30; until 70 epochs")

            # NOTE: 20240813: decided to use very small number of training material
            # V1: 10% of original = 0.001
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
        else: 
            torch.cuda.set_device(args.gpu)
            runnumber = args.runnumber
            # model_types = ['large', 'reslin', 'lstm']
            for preepoch in [0, 15]: # 10, 15, 20, 25, 30, 0, , 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60
            # for model_type in model_types: 
                print(f"Model {args.model}, PreEpoch {preepoch}, PreType {args.pretype}")
                if preepoch == 0 and args.pretype != "l": 
                    print("Skip hf for 0 preepoch")
                    continue
                run_once(model_save_dir, model_type=args.model, pretype=args.pretype, posttype="f", sel=args.select, 
                         preepochs=preepoch, postepochs=(30 - preepoch), save_model=False)
