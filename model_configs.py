class ModelDimConfigs: 
    def __init__(self,  
                 rnn_in_size, 
                 lin_in_size_1, 
                 lin_in_size_2, 
                 lin_out_size_2, 
                 ): 
        
        self.model_in_size = rnn_in_size
        self.model_out_size = lin_out_size_2


        self.rnn_in_size = rnn_in_size
        self.rnn_out_size = lin_in_size_1
        self.lin_in_size_1 = lin_in_size_1
        self.lin_out_size_1 = lin_in_size_2
        self.lin_in_size_2 = lin_in_size_2
        self.lin_out_size_2 = lin_out_size_2


class TrainingConfigs: 
    BATCH_SIZE = 256

    REC_SAMPLE_RATE = 16000
    N_FFT = 400
    N_MELS = 64
    
    LOADER_WORKER = 16
