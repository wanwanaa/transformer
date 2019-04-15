from models import *


def load_model(config, filename):
    model = Transformer(config)
    model.load_state_dict(torch.load(filename, map_location='cpu'))
    return model


def save_model(model, filename):
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), filename)
    else:
        torch.save(model.state_dict(), filename)
    print('model save at ', filename)