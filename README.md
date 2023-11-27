# lffl
Low to Full Frequency Learning (lffl)



## How to set up enviroment
1. Create directory and its subdirectory "script"
2. Put all files in repo into script
3. use conda env create -f environment.yml to set up environment, if you have anaconda
4. run python paths.py to set up src directories
5. preproc_ prefixed files are used in preprocessing
6. misc_ prefixed files are other utils
7. debug_ prefixed files are for development


## Preprocessing
1. preproc_seg.py: run this file and get the continuous recordings cut into phones  
2. preproc_guide_integrate.py: run and integrate the guide files into one large guideline  
3. preproc_guide_mod.py: run and make additional changes to the guide. You can self-define any change because this is post-hoc  
- preproc_guide_extract.py: use it if you only want to exract the metadata but not touch the recordings
4. preproc_guide_addpath.py (optional): add extra path combined from rec and idx. This will take around twice the size in storage but will save calculation time when loading dataset. 
5. preproc_guide_sepTVT.py: separate training, validation as well as test dataset. This will make sure any speaker (not only segments) is only in one of the sets. 



## Frank's Notes
1. The phonetic aligment (&transcription) is using [ARPABET](https://en.wikipedia.org/wiki/ARPABET), with alphabet (combination)s marking sounds and numbers noting stress. 

2. We can just leave the structure of the dataset as it is after cutting. Since it will just change the path of files, it won't really affect the reading efficiency during training. 

3. Ideaology for combining small files into one:   
    - Loading time using pickle: 55.0732 seconds
    - Loading time from individual files: 181.1897 seconds
    - This poses the question of whether we really need to cut them (😢) -> not really that important though. Let's stick to cutting. (Because we have multiprocessing during reading)
4. aaa