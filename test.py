from models import *
from utils import *
import pickle


if __name__ == '__main__':
    config = Config()
    filename = config.filename_model + 'model_0.pkl'
    model = load_model(config, filename)

    # data
    test_loader = data_load(config.filename_trimmed_test, config.batch_size, False)
    f = open(config.filename_idx2word, 'rb')
    idx2word = pickle.load(f)

    for batch in test_loader:
        x, y = batch
        x_pos = torch.arange(config.max_len).repeat(x.size(0), 1)
        y_pos = torch.arange(config.max_len).repeat(x.size(0), 1)
        result, _ = model.sample(x, x_pos, y, y_pos)

        sen = index2sentence(list(result[-1]), idx2word)
        sen = ' '.join(sen)
        print(sen)