# Here we put some functions / classes as tool
# Let's make the code better structured! 
import os

class PathUtils: 
    @staticmethod
    def path_exist(path): 
        return os.path.exists(path)

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