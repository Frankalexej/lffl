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
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, silhouette_score, adjusted_rand_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
from torchinfo import summary
import torch.nn.functional as F
from torch.nn import init
from A_00_models import CNNAutoencoder, ResLinearAutoencoder, LSTMAutoencoder
from model_configs import ModelDimConfigs, TrainingConfigs
from misc_tools import get_timestamp, ARPABET
from model_dataset import DS_Tools, Padder, TokenMap, NormalizerKeepShapeManual
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

def draw_learning_curve_and_accuracy_new(data, type_names, task_names, epoch="", best_val=None, save=False, save_name=""): 
    plt.clf()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for task_id, task_name in enumerate(task_name): 
        ax = axes[task_id // 2, task_id % 2]
        for type_id, type_name in enumerate(type_names): 
            values = data[type_id][task_id]
            ax.plot(values, label=type_name)
        ax.set_title(f'Learning Curve {task_name}' + f" {epoch}")

    # Display the plots
    plt.tight_layout()
    plt.xlabel("Epoch")
    display.clear_output(wait=True)
    display.display(plt.gcf())
    if save: 
        plt.savefig(save_name)

def draw_learning_curve_and_accuracy(accs, epoch="", best_val=None, save=False, save_name=""): 
    plt.clf()
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 4))
    valid_accs, full_valid_accs = accs

    # Plot Accuracy on the right subplot
    ax1.plot(valid_accs, label='Valid')
    ax1.plot(full_valid_accs, label='Full Valid')
    ax1.set_title('Learning Curve Silhouette Score' + f" {epoch}")
    ax1.legend(loc="lower right")

    # Display the plots
    plt.tight_layout()
    plt.xlabel("Epoch")
    display.clear_output(wait=True)
    display.display(plt.gcf())
    if save: 
        plt.savefig(save_name)

def clustering_accuracy(y_true, y_pred):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Use the Hungarian algorithm to find the best assignment
    row_ind, col_ind = linear_sum_assignment(-cm)
    # Calculate the accuracy
    accuracy = cm[row_ind, col_ind].sum() / y_true.size
    return accuracy

def kmeans_evaluate(X, Y, n_clusters=50, n_init=20, random_state=0, epoch=0, model_save_dir="", name=""):
    if n_clusters is None:
        n_clusters = len(np.unique(Y))
    # X_std = StandardScaler().fit_transform(X)
    # pca = PCA(n_components=96, random_state=random_state)
    # X_pca = pca.fit_transform(X_std)
    # # X_pca = X
    # km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state, 
    #             max_iter=200, algorithm="elkan", init="k-means++", tol=1e-3)
    # cluster_ids = km.fit_predict(X_pca)
    # acc = clustering_accuracy(Y, cluster_ids)
    acc = 0
    sil = silhouette_score(X, Y)
    # sil = 0
    # ari = adjusted_rand_score(Y, cluster_ids)
    ari = 0
    dbi = davies_bouldin_score(X, Y)

    # np.save(os.path.join(model_save_dir, f"{epoch:04d}_{name}_y_pred.npy"), cluster_ids)
    return {
        "kmeans_acc": acc,
        "silhouette": sil,
        "adjusted_rand_index": ari, 
        "davies_bouldin_score": dbi
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
    
    def read_eval_func(self, name, epoch): 
        z = np.load(os.path.join(self.model_save_dir, f"{epoch:04d}_{name}_z.npy"))
        y = np.load(os.path.join(self.model_save_dir, f"{epoch:04d}_{name}_y.npy"))
        return z, y

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

    valid_accs = ListRecorder(os.path.join(model_save_dir, "valid.ncacc"))
    full_valid_accs = ListRecorder(os.path.join(model_save_dir, "full_valid.ncacc"))

    valid_sils = ListRecorder(os.path.join(model_save_dir, "valid.ncsil"))
    full_valid_sils = ListRecorder(os.path.join(model_save_dir, "full_valid.ncsil"))

    valid_aris = ListRecorder(os.path.join(model_save_dir, "valid.ncari"))
    full_valid_aris = ListRecorder(os.path.join(model_save_dir, "full_valid.ncari"))

    valid_dbis = ListRecorder(os.path.join(model_save_dir, "valid.ncdbi"))
    full_valid_dbis = ListRecorder(os.path.join(model_save_dir, "full_valid.ncdbi"))

    special_recs = DictRecorder(os.path.join(model_save_dir, "special.hst"))

    eval_saver = EvalSaver(model_save_dir)

    # Initialize Model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # criterion = nn.MSELoss()
    # input_shape = (128, 1, 64, 32)
    # if model_type == "cnn": 
    #     model = CNNAutoencoder(input_shape=input_shape)
    # elif model_type == "reslin": 
    #     model = ResLinearAutoencoder(input_shape=input_shape)
    # elif model_type == "lstm": 
    #     model = LSTMAutoencoder()
    # else:
    #     raise Exception("Model not defined! ")
    # model.to(device)

    # Load Data (I&II)
    # train_loader_1 = load_data(type=pretype, sel="full", load="train")
    # valid_loader_1 = load_data(type=pretype, sel=sel, load="valid") # target 
    # train_loader_2 = load_data(type=posttype, sel="full", load="train")
    # valid_loader_2 = load_data(type=posttype, sel=sel, load="valid")    # full = trainlike (because this time we don't separate c/v)

    """No Learning Baseline Get"""
    valid_z, valid_y = eval_saver.read_eval_func("valid", 9999) # to make sure the file is there.
    full_valid_z, full_valid_y = eval_saver.read_eval_func("full_valid", 9999)
    # Target Eval
    # model.eval()
    # valid_loss = 0.
    # valid_num = len(valid_loader_1)
    # z_list, y_list = [], []
    # for idx, (x, y) in enumerate(valid_loader_1):
    #     # NOTE: still, x is data, y is label. But instead we will output x_hat, not y_hat. 
    #     x = x.to(device)
    #     y = y.to(device)

    #     x_hat, z = model(x, return_latent=True)

    #     z_list.append(z.detach().cpu().numpy())
    #     y_list.append(y.detach().cpu().numpy())

    # z_all, y_all = concat_func(z_list, y_list)
    # eval_saver.save_eval_func(valid_z, valid_y, "valid", 9999, save_eval)
    res = kmeans_evaluate(valid_z, valid_y, n_clusters=50, epoch=9999, model_save_dir=model_save_dir, name="valid")
    valid_accs.append(res["kmeans_acc"])
    valid_sils.append(res["silhouette"])
    valid_aris.append(res["adjusted_rand_index"])
    valid_dbis.append(res["davies_bouldin_score"])

    # Full Eval
    # model.eval()
    # full_valid_loss = 0.0
    # full_valid_num = len(valid_loader_2)
    # z_list, y_list = [], []
    # for idx, (x, y) in enumerate(valid_loader_2):
    #     x = x.to(device)
    #     y = y.to(device)

    #     x_hat, z = model(x, return_latent=True)
        
    #     z_list.append(z.detach().cpu().numpy())
    #     y_list.append(y.detach().cpu().numpy())

    # z_all, y_all = concat_func(z_list, y_list)
    # eval_saver.save_eval_func(z_all, y_all, "full_valid", 9999, save_eval)
    res = kmeans_evaluate(full_valid_z, full_valid_y, n_clusters=50, 
                          epoch=9999, model_save_dir=model_save_dir, name="full_valid")
    full_valid_accs.append(res["kmeans_acc"])
    full_valid_sils.append(res["silhouette"])
    full_valid_aris.append(res["adjusted_rand_index"])
    full_valid_dbis.append(res["davies_bouldin_score"])

    valid_accs.save()
    full_valid_accs.save()
    valid_sils.save()
    full_valid_sils.save()
    valid_aris.save()
    full_valid_aris.save()
    valid_dbis.save()
    full_valid_dbis.save()

    # Train (I)
    best_valid_loss = 1e9
    best_valid_loss_epoch = 0
    BASE = 0

    for epoch in range(BASE, BASE + preepochs):
        print(f"Epoch {epoch}")
        valid_z, valid_y = eval_saver.read_eval_func("valid", epoch) # to make sure the file is there.
        full_valid_z, full_valid_y = eval_saver.read_eval_func("full_valid", epoch)

        valid_res = kmeans_evaluate(valid_z, valid_y, n_clusters=50, 
                                    epoch=epoch, model_save_dir=model_save_dir, name="valid")
        full_valid_res = kmeans_evaluate(full_valid_z, full_valid_y, n_clusters=50, 
                                         epoch=epoch, model_save_dir=model_save_dir, name="full_valid")

        valid_accs.append(valid_res["kmeans_acc"])
        valid_sils.append(valid_res["silhouette"])
        valid_aris.append(valid_res["adjusted_rand_index"])
        valid_dbis.append(valid_res["davies_bouldin_score"])
        full_valid_accs.append(full_valid_res["kmeans_acc"])
        full_valid_sils.append(full_valid_res["silhouette"])
        full_valid_aris.append(full_valid_res["adjusted_rand_index"])
        full_valid_dbis.append(full_valid_res["davies_bouldin_score"])
        valid_accs.save()
        full_valid_accs.save()
        valid_sils.save()
        full_valid_sils.save()
        valid_aris.save()
        full_valid_aris.save()
        valid_dbis.save()
        full_valid_dbis.save()

    # Train (II)
    BASE = BASE + preepochs
    real_postepochs = 30 - preepochs
    for epoch in range(BASE, BASE + real_postepochs):
        print(f"Epoch {epoch}")
        valid_z, valid_y = eval_saver.read_eval_func("valid", epoch) # to make sure the file is there.
        full_valid_z, full_valid_y = eval_saver.read_eval_func("valid", epoch)

        valid_res = kmeans_evaluate(valid_z, valid_y, n_clusters=50, 
                                    epoch=epoch, model_save_dir=model_save_dir, name="valid")
        full_valid_res = kmeans_evaluate(full_valid_z, full_valid_y, n_clusters=50, 
                                         epoch=epoch, model_save_dir=model_save_dir, name="full_valid")

        valid_accs.append(valid_res["kmeans_acc"])
        valid_sils.append(valid_res["silhouette"])
        valid_aris.append(valid_res["adjusted_rand_index"])
        valid_dbis.append(valid_res["davies_bouldin_score"])
        full_valid_accs.append(full_valid_res["kmeans_acc"])
        full_valid_sils.append(full_valid_res["silhouette"])
        full_valid_aris.append(full_valid_res["adjusted_rand_index"])
        full_valid_dbis.append(full_valid_res["davies_bouldin_score"])
        valid_accs.save()
        full_valid_accs.save()
        valid_sils.save()
        full_valid_sils.save()
        valid_aris.save()
        full_valid_aris.save()
        valid_dbis.save()
        full_valid_dbis.save()

    # draw_learning_curve_and_accuracy_new(data=[[valid_accs.get(), full_valid_accs.get()],
    #                                            [valid_sils.get(), full_valid_sils.get()],
    #                                               [valid_aris.get(), full_valid_aris.get()]],
    #                                 task_names=["KmeansAcc", "Silhouette", "AdjRandIdx"],
    #                                 type_names=["Valid", "FullValid"],
    #                                 epoch=str(BASE + postepochs - 1), 
    #                                 save=True, 
    #                                 save_name=f"{model_save_dir}/vis_acc.png")
    
    # draw_learning_curve_and_accuracy(accs=(valid_dbis.get(), full_valid_dbis.get()),
    #                                 epoch=str(BASE + real_postepochs - 1),
    #                                 save=True,
    #                                 save_name=f"{model_save_dir}/vis_dbi.png")

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
            for preepoch in [15]: # 10, 15, 20, 25, 30, 0, , 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60
                print(f"Model {args.model}, PreEpoch {preepoch}, PreType {args.pretype}")
            # for model_type in model_types: 
                # if preepoch == 0 and args.pretype != "l": 
                #     print("Skip hf for 0 preepoch")
                #     continue
                run_once(model_save_dir, model_type=args.model, pretype=args.pretype, posttype="f", sel=args.select, 
                            preepochs=preepoch, postepochs=(120 - preepoch), save_model=False)
