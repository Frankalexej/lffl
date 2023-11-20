import pickle

import matplotlib.pylab as plt
from IPython import display
import pandas as pd
import numpy as np

# Define recorders of training hists, for ease of extension
class Recorder: 
    def __init__(self, IOPath): 
        self.record = []
        self.IOPath = IOPath

    def save(self): 
        pass
    
    def append(self, content): 
        self.record.append(content)
    
    def get(self): 
        return self.record
    

class ListRecorder(Recorder): 
    def read(self): 
        # only used by loss hists 
        with open(self.IOPath, 'rb') as f:
            self.record = pickle.load(f)
    
    def save(self): 
        with open(self.IOPath, 'wb') as file:
            pickle.dump(self.record, file)


class DfRecorder(): 
    def __init__(self, IOPath): 
        self.record = pd.DataFrame()
        self.IOPath = IOPath
    def read(self): 
        self.record = pd.read_csv(self.IOPath)
    
    def save(self): 
        self.record.to_csv(self.IOPath, index=False)

    def append(self, content): 
        self.record = pd.concat([self.record, pd.DataFrame([content])], ignore_index=True)

    def get(self): 
        return self.__convert_lists_to_arrays(self.record.to_dict('list'))
    
    @staticmethod
    def __convert_lists_to_arrays(input_dict):
        """
        Convert all lists in a dictionary to NumPy arrays.

        Args:
        - input_dict (dict): Dictionary with lists as values.

        Returns:
        - dict: Dictionary with lists converted to NumPy arrays.
        """
        output_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, list):
                output_dict[key] = np.array(value)
            elif isinstance(value, dict):
                output_dict[key] = DfRecorder.__convert_lists_to_arrays(value)
            else:
                output_dict[key] = value
        return output_dict

class HistRecorder(Recorder):     
    def save(self): 
        with open(self.IOPath, "a") as txt:
            txt.write("\n".join(self.record))
    
    def print(self, content): 
        self.append(content)
        print(content)

def draw_learning_curve(train_losses, valid_losses, title="Learning Curve Loss", epoch=""): 
    plt.clf()
    plt.plot(train_losses, label='Train')
    plt.plot(valid_losses, label='Valid')
    plt.title(title + f" {epoch}")
    plt.legend(loc="upper right")
    display.clear_output(wait=True)
    display.display(plt.gcf())


def draw_learning_curve_and_accuracy(losses, accs, epoch=""): 
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    train_losses, valid_losses, best_val_loss = losses
    valid_precs, valid_recs, valid_fs = accs

    # Plot Loss on the left subplot
    ax1.plot(train_losses, label='Train')
    ax1.plot(valid_losses, label='Valid')
    ax1.axvline(x=best_val_loss, color='r', linestyle='--', label=f'Best: {best_val_loss}')
    ax1.set_title("Learning Curve Loss" + f" {epoch}")
    ax1.legend(loc="upper right")

    # Plot Accuracy on the right subplot
    ax2.plot(valid_precs, label='Precision')
    ax2.plot(valid_recs, label='Recall')
    ax2.plot(valid_fs, label='F1-score')
    ax2.set_title('Learning Curve Accuracy' + f" {epoch}")
    ax2.legend(loc="lower right")

    # Display the plots
    plt.tight_layout()
    plt.xlabel("Epoch")
    display.clear_output(wait=True)
    display.display(plt.gcf())

def save_learning_curve_and_accuracy(losses, accs, epoch="", save_name=""): 
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    train_losses, valid_losses, best_val_loss = losses
    valid_precs, valid_recs, valid_fs = accs

    # Plot Loss on the left subplot
    ax1.plot(train_losses, label='Train')
    ax1.plot(valid_losses, label='Valid')
    ax1.axvline(x=best_val_loss, color='r', linestyle='--', label=f'Best: {best_val_loss}')
    ax1.set_title("Learning Curve Loss" + f" {epoch}")
    ax1.legend(loc="upper right")

    # Plot Accuracy on the right subplot
    ax2.plot(valid_precs, label='Precision')
    ax2.plot(valid_recs, label='Recall')
    ax2.plot(valid_fs, label='F1-score')
    ax2.axhline(y=valid_precs[best_val_loss], color='r', linestyle='--', label=f'Best: {valid_precs[best_val_loss]}')
    ax2.set_title('Learning Curve Accuracy' + f" {epoch}")
    ax2.legend(loc="lower right")

    # Display the plots
    plt.tight_layout()
    plt.xlabel("Epoch")
    plt.savefig(save_name)

def draw_learning_curve_and_dissimilarity(losses, dissimilarities, epoch="", save=False, save_name=""): 
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    train_losses, valid_losses, best_val_loss = losses
    same_dict, diff_dict = dissimilarities

    # Plot Loss on the left subplot
    ax1.plot(train_losses, label='Train')
    ax1.plot(valid_losses, label='Valid')
    ax1.axvline(x=best_val_loss, color='r', linestyle='--', label=f'Best: {best_val_loss}')
    ax1.set_title("Learning Curve Loss" + f" {epoch}")
    ax1.legend(loc="upper right")

    # Plot Accuracy on the right subplot
    time_steps = np.arange(len(same_dict["mean"]))
    ax2.errorbar(time_steps, same_dict["mean"], yerr=(same_dict["mean"] - same_dict["ci_lower"], same_dict["ci_upper"] - same_dict["mean"]), label='Same', fmt='o-', capsize=5)
    ax2.errorbar(time_steps, diff_dict["mean"], yerr=(diff_dict["mean"] - diff_dict["ci_lower"], diff_dict["ci_upper"] - diff_dict["mean"]), label='Different', fmt='o-', capsize=5)
    ax2.set_title('Learning Curve Dissimilarity' + f" {epoch}")
    ax2.legend(loc="lower right")

    # Display the plots
    plt.tight_layout()
    plt.xlabel("Epoch")
    display.clear_output(wait=True)
    display.display(plt.gcf())
    if save: 
        plt.savefig(save_name)