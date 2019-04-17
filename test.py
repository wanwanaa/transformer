from models import *
from train import *
from utils import *

if __name__ == '__main__':
    config = Config()
    # vocab = Vocab(config)
    filename = config.filename_model + 'model_0.pkl'
    model = load_model(config, filename)
    loss_func = LabelSmoothing(config)
    test(0, config, model, loss_func)
