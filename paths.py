import os

root_ = "../"   # hyperpath of here
src_ = root_ + "/src/"
train_audio_ = src_ + "train-clean-100-audio/"
train_tg_ = src_ + "train-clean-100-tg/"
train_cut_audio_ = src_ + "train-clean-100-ca/"  # cut audio, ideally following structure of original audio
dev_cut_audio_ = src_ + "dev-clean-ca/"



def mk(dir): 
    os.makedirs(dir, exist_ok = True)


if __name__ == '__main__':
    # For all paths defined, run mk()
    for name, value in globals().copy().items():
        if isinstance(value, str) and not name.startswith("__"):
            globals()[name] = mk(value)