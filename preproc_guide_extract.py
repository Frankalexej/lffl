from praatio import textgrid
import pandas as pd
import os

from paths import *
from misc_tools import PathUtils as PU
from misc_progress_bar import draw_progress_bar


def extract_from_tg(read_path, save_dir, tg_name, save_small=True): 
    tg = textgrid.openTextgrid(read_path, False)
    entries = tg.getTier("phones").entries # Get all intervals

    segment = []    # note down what sound it is
    file = []       # filename of [sentence], not sound
    id = []         # address within sentence file
    startTime = []  # start time of segment
    endTime = []    # end time of segment

    for idx, segment_interval in enumerate(entries): 
        segment.append(segment_interval.label)
        file.append(tg_name)
        id.append(idx)
        startTime.append(segment_interval.start)
        endTime.append(segment_interval.end)
    

    data = {
    'segment': segment,
    'file': file,
    'id': id, 
    'startTime': startTime,
    'endTime': endTime
    }

    if save_small: 
        # Create a Pandas DataFrame
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(save_dir, tg_name + ".csv"), index=False)
    return data


def extract_from_tgs(src_path, target_path): 
    """
    structure of src_path example: 
    train-clean-100-tg/19/198/19-198-0000.TextGrid
    src_path is the hyperpath of all textgrids
    make sure target_path is existent
    """
    assert (PU.path_exist(target_path) and PU.path_exist(src_path))

    # total_data_bucket = []
    total_speakers = len(os.listdir(src_path))

    for idx, speaker_ in enumerate(sorted(os.listdir(src_path), key=str.casefold)): 
        # train-clean-100-tg/[19]/198/19-198-0000.TextGrid
        src_speaker_ = os.path.join(src_path, speaker_)
        if not os.path.isdir(src_speaker_): 
            continue
        tgt_speaker_ = os.path.join(target_path, speaker_)
        PU.mk(tgt_speaker_)
        for rec_ in sorted(os.listdir(src_speaker_), key=str.casefold): 
            src_speaker_rec_ = os.path.join(src_speaker_, rec_)
            tgt_speaker_rec_ = os.path.join(tgt_speaker_, rec_)
            PU.mk(tgt_speaker_rec_)
            for sentence in sorted(os.listdir(src_speaker_rec_), key=str.casefold): 
                # here we loop through each textgrid file
                data = extract_from_tg(
                    read_path=os.path.join(src_speaker_rec_, sentence), 
                    save_dir=tgt_speaker_rec_, 
                    tg_name=os.path.splitext(sentence)[0]
                )
                # total_data_bucket.append(data)
        draw_progress_bar(idx, total_speakers)
    
    # total_df = pd.DataFrame(total_data_bucket)
    # total_df.to_csv(os.path.join(src_path, "train_guide.csv"), index=False)
    return



if __name__ == "__main__": 
    extract_from_tgs(train_tg_, train_cut_audio_)