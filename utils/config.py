class Config():
    def __init__(self):
        # data
        self.filename_trimmed_train = 'data/valid.pt'
        self.filename_trimmed_valid = 'data/valid.pt'
        self.filename_trimmed_test = 'data/test.pt'

        # vocab
        self.filename_word2idx = 'DATA/data/word2index.pkl'
        self.filename_idx2word = 'DATA/data/index2word.pkl'

        # Hyper Parameters
        self.LR = 0.0003
        self.batch_size = 256
        self.iters = 10000
        self.len = 150
        self.beam_size = 10
        self.pad = 0

        self.vocab_size = 4000
        self.model_size = 512
        self.d_ff = 1024
        self.dropout = 0.1
        self.n_head = 8
        self.n_layer = 6