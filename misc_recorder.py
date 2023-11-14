import pickle

import matplotlib.pylab as plt
from IPython import display

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