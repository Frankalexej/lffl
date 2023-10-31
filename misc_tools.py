# Here we put some functions / classes as tool
# Let's make the code better structured! 
import os
import datetime
import pandas as pd
from IPython.display import Audio, display

class PathUtils: 
    @staticmethod
    def path_exist(path): 
        return os.path.exists(path)
    
    @staticmethod
    def path_isdir(path): 
        return os.path.isdir(path)

    @staticmethod
    def mk(dir): 
        os.makedirs(dir, exist_ok = True)



class ARPABET: 
    @staticmethod
    def is_vowel(arpabet_transcription):
        vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY', 'EH', 'ER', 'EY', 'IH', 'IX', 'IY', 'OW', 'OY', 'UH', 'UW', 'UX']
        
        if arpabet_transcription in vowels:
            return True
        else:
            return False
    
    @staticmethod
    def is_consonant(arpabet_transcription):
        consonants = [
            'B', 'CH', 'D', 'DH', 'DX', 'EL', 'EM', 'EN', 'F', 'G', 'HH', 'H', 'JH', 'K', 'L', 'M', 'N',
            'NX', 'NG', 'P', 'Q', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'WH', 'Y', 'Z', 'ZH'
        ]
        
        if arpabet_transcription in consonants:
            return True
        else:
            return False
    
    @staticmethod
    def vowel_consonant(arpabet_transcription): 
        if ARPABET.is_vowel(arpabet_transcription): 
            return "vowel"
        elif ARPABET.is_consonant(arpabet_transcription): 
            return "consonant"
        else: 
            return "nap"
        
class MyAudio: 
    @staticmethod
    def play_audio_torch(waveform, sample_rate):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        if num_channels == 1:
            display(Audio(waveform[0], rate=sample_rate))
        elif num_channels == 2:
            display(Audio((waveform[0], waveform[1]), rate=sample_rate))
        else:
            raise ValueError("Waveform with more than 2 channels are not supported.")
        
    @staticmethod
    def play_audio_np(waveform, sample_rate):
        display(Audio(waveform[0], rate=sample_rate))


class AudioCut: 
    @staticmethod
    def time2frame(time_sec, rate): 
        # Calculate the number of elapsed samples
        return int(round(time_sec * rate))
    
    @staticmethod
    def idx2text(idx, fill_num): 
        return str(idx).zfill(fill_num)
    
    @staticmethod
    def cut_name_gen(name, idx, fill_num, bare=False): 
        if bare: 
            return name + "-" + AudioCut.idx2text(idx, fill_num)
        return name + "-" + AudioCut.idx2text(idx, fill_num) + ".flac"
    
    @staticmethod
    def solve_name(name): 
        # bare name, no extension
        # 19-198-0037 (example)
        return tuple(name.split("-"))

    @staticmethod
    def record2filepath(record): 
        # record is a row
        sentence = record["file"]
        idx = AudioCut.idx2text(record["id"], fill_num=4)

        return sentence.replace("-", "/") + "/" + sentence + "-" + idx + ".flac"

# START
def get_timestamp():
    # for model save
    return datetime.datetime.now().strftime("%m%d%H%M%S")   # timestamp ignores year and second, not really needed. 