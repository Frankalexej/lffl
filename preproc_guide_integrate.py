import pandas as pd
import os

from paths import *
from ssd_paths import *
from misc_tools import PathUtils as PU
from misc_progress_bar import draw_progress_bar

def remove_stress(segment):
    if segment and segment[-1].isdigit():
        return segment[:-1]
    return segment

def integrate_guides(src_path, target_filename): 
    """
    structure of src_path example: 
    train-clean-100-tg/19/198/19-198-0000.TextGrid
    src_path is the hyperpath of all textgrids
    make sure target_path is existent
    """
    assert (PU.path_exist(src_path))

    total_df = pd.DataFrame()
    total_speakers = len(os.listdir(src_path))

    for idx, speaker_ in enumerate(sorted(os.listdir(src_path), key=str.casefold)): 
        # train-clean-100-tg/[19]/198/19-198-0000.TextGrid
        src_speaker_ = os.path.join(src_path, speaker_)
        if not os.path.isdir(src_speaker_): 
            continue
        for rec_ in sorted(os.listdir(src_speaker_), key=str.casefold): 
            src_speaker_rec_ = os.path.join(src_speaker_, rec_)
            for sentence in sorted(os.listdir(src_speaker_rec_), key=str.casefold): 
                # here we loop through each csv guide file
                small_guide_df = pd.read_csv(os.path.join(src_speaker_rec_, sentence))
                total_df = pd.concat([total_df, small_guide_df], ignore_index=True)
        draw_progress_bar(idx, total_speakers)
    
    # post-hoc changes
    # total_df['segment_nostress'] = total_df['segment'].apply(remove_stress)

    total_df.to_csv(target_filename, index=False)
    return



if __name__ == "__main__": 
    # integrate_guides(train_cut_guide_, os.path.join(src_, "guide.csv"))
    integrate_guides(strain_cut_guide_, os.path.join(ssrc_, "guide.csv"))