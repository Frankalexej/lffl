"""
Other modifications on generated guide
"""
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

def stress_value(segment): 
    if segment and segment[-1].isdigit(): 
        return segment[-1]  # this is also str, not int
    return "SNA" # meaning stress not applicable

def modify_guide(src_path, target_path): 
    assert (PU.path_exist(src_path))

    total_df = pd.read_csv(src_path)
    # post-hoc changes
    total_df['segment_nostress'] = total_df['segment'].apply(remove_stress)
    total_df['stress_type'] = total_df['segment'].apply(stress_value)

    total_df.to_csv(target_path, index=False)
    return



if __name__ == "__main__": 
    # modify_guide(os.path.join(src_, "guide.csv"), os.path.join(src_, "guide_mod.csv"))
    modify_guide(os.path.join(ssrc_, "guide.csv"), os.path.join(ssrc_, "guide_mod.csv"))