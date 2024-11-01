# Non-full to Full Frequency Learning




## How to set up enviroment
1. Create directory and download the repo
2. For viewing purposes we separated "preprocessing", "configuration", "model" and "running" directories, but when they were written and run they were all under "scripts" directory. Please create a "scripts" under the main directory and put all scripts there. This will not create confusion or collision since they have distinct prefixes. 
3. use conda env create -f environment.yml to set up environment, if you have anaconda
4. run python paths.py to set up src directories and put audio and annotation data there accordingly
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

## Models
1. The latest model definitions could be found under model/H_10_models.py
2. Datasets are under model/model_dataset.py
3. model/model_filter.py contains the codes for frequency filtering

# Running
1. H_20_all.py is for experiment 1
2. H_21_stage.py is for experiment 2
3. H_22_information_pretest.ipynb is for testing the low frequency and high frequency information tests

# Analysis
1. Analysis is done separately using R scripts
2. The source data were organized and outputed using running/G_20 and G_21 into csv



# Notes
1. The phonetic aligment (&transcription) is using [ARPABET](https://en.wikipedia.org/wiki/ARPABET), with alphabet (combination)s marking sounds and numbers noting stress. 

2. We can just leave the structure of the dataset as it is after cutting. Since it will just change the path of files, it won't really affect the reading efficiency during training. 

3. Ideology for combining small files into one:   
    - Loading time using pickle: 55.0732 seconds
    - Loading time from individual files: 181.1897 seconds
    - This poses the question of whether we really need to cut them (😢) -> not really that important though. Let's stick to cutting. (Because we have multiprocessing during reading)