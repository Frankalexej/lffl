import os

sroot_ = "/media/ldlmdl/A2AAE4B1AAE482E1/SSD_Documents/lffl/"   # hyperpath of here
ssrc_ = sroot_ + "/src/"
strain_audio_ = ssrc_ + "train-clean-100-audio/"
strain_tg_ = ssrc_ + "train-clean-100-tg/"
strain_cut_audio_ = ssrc_ + "train-clean-100-ca/"  # cut audio, ideally following structure of original audio
strain_cut_guide_ = ssrc_ + "train-clean-100-cg/"  # cut guide, ideally following structure of original audio


def mk(dir): 
    os.makedirs(dir, exist_ok = True)


if __name__ == '__main__':
    # For all paths defined, run mk()
    for name, value in globals().copy().items():
        if isinstance(value, str) and not name.startswith("__"):
            globals()[name] = mk(value)