import os

root_ = "../"   # hyperpath of here
src_ = root_ + "/src/"
train_audio_ = src_ + "train-clean-100-audio/"
train_tg_ = src_ + "train-clean-100-tg/"
train_cut_audio_ = src_ + "train-clean-100-ca/"  # cut audio, ideally following structure of original audio
train_cut_guide_ = src_ + "train-clean-100-cg/"  # cut guide, ideally following structure of original audio

# # trial paths
# try_audio_ = src_ + "try-clean-audio/"
# try_tg_ = src_ + "try-clean-tg/"
# try_cut_audio_ = src_ + "try-clean-ca/"  # cut audio, ideally following structure of original audio
# try_cut_guide_ = src_ + "try-clean-cg/"  # cut guide, ideally following structure of original audio

debug_ = src_ + "debug/"

model_save_ = root_ + "model_save/"

use_ = src_ + "/use/"



def mk(dir): 
    os.makedirs(dir, exist_ok = True)


if __name__ == '__main__':
    # For all paths defined, run mk()
    for name, value in globals().copy().items():
        if isinstance(value, str) and not name.startswith("__"):
            globals()[name] = mk(value)