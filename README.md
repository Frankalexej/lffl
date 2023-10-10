# lffl
Low to Full Frequency Learning (lffl)



## How to set up enviroment
1. Create directory and its subdirectory "script"
2. Put all files in repo into script
3. use conda env create -f environment.yml to set up environment, if you have anaconda
4. run python paths.py to set up src directories
5. preproc_ prefixed files are used in preprocessing
6. misc_ prefixed files are other utils
7. aaa



## Frank's Notes
1. The phonetic aligment (&transcription) is using [ARPABET](https://en.wikipedia.org/wiki/ARPABET), with alphabet (combination)s marking sounds and numbers noting stress. 

2. We can just leave the structure of the dataset as it is after cutting. Since it will just change the path of files, it won't really affect the reading efficiency during training. 