class Config():
    def __init__(self):
        # data
        self.filename_trimmed_train = 'data/clean/data_hybird/valid.pt'
        self.filename_trimmed_valid = 'data/clean/data_hybird/valid.pt'
        self.filename_trimmed_test = 'data/clean/data_hybird/test.pt'

        # vocab
        # self.filename_word2idx = 'data/clean/data_char/src_word2index.pkl'
        self.filename_idx2word = 'data/clean/data_hybird/tgt_index2word.pkl'
        # self.filename_word2idx = 'data/clean/data_hybird/src_word2index.pkl'

        # filename result
        #############################################
        self.filename_data = 'result/data/'
        self.filename_model = 'result/model/'
        self.filename_rouge = 'result/data/ROUGE.txt'
        #############################################
        self.filename_gold = 'result/gold/gold_summaries.txt'

        # Hyper Parameters
        self.LR = 0.003
        self.batch_size = 64
        self.iters = 10000

        self.t_len = 80
        self.s_len = 50

        self.beam_size = 10
        self.pad = 0
        self.bos = 2

        # self.src_vocab_size = 4000
        # self.tgt_vocab_size = 4000
        self.src_vocab_size = 523566
        self.tgt_vocab_size = 8250
        self.share_vocab = False
        self.model_size = 512
        self.d_ff = 1024
        self.dropout = 0.1
        self.n_head = 8
        self.n_layer = 6

        # self.optimizer = torch.optim.Adam()
        self.warmup_steps = 4000

        self.ls = 0.1
        self.accumulation_steps = 8