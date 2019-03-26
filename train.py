import torch
import argparse
from models import *
from utils import *


def train(args, config, model):
    model.train()
    # optim
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    optim = Optim(optimizer, config)

    # data
    train_loader = data_load(config.filename_trimmed_train, config.batch_size, True)

    # loss result
    train_loss = []
    valid_loss = []
    test_loss = []
    test_rouge = []

    for e in range(args.epoch):
        model.train()
        all_loss = 0
        num = 0
        for step, batch in enumerate(train_loader):
            x, y = batch
            x_pos = torch.arange(config.t_len).repeat(x.size(0), 1)
            y_pos = torch.arange(config.s_len).repeat(x.size(0), 1)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
                x_pos = x.pos.cuda()
                y_pos = y_pos.cuda()

            out, loss = model(x, x_pos, y, y_pos)

            optim.zero_grad()
            loss.backward()
            optim.updata()

            n_word = y.ne(config.pad).sum().item()
            num += n_word

            all_loss += loss.item()
            if step % 200 == 0:
                print('epoch:', e, '|step:', step, '|train_loss: %.4f' % loss.item())

        # train loss
        loss = all_loss / num
        print('epoch:', e, '|train_loss: %.4f' % loss)
        train_loss.append(loss)


def main():
    config = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size for train')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='number of training epochs')
    parser.add_argument('--n_layer', '-n', type=int, default=6, help='number of encoder layers')
    parser.add_argument('-seed', '-s', type=int, default=123, help="Random seed")
    parser.add_argument('--save_model', '-m', action='store_true', default=False, help="whether to save model")
    args = parser.parse_args()

    ########test##########
    args.batch_size = 2
    ########test##########

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.n_layer:
        config.n_layer = args.n_layer

    # seed
    torch.manual_seed(args.seed)

    # rouge initalization
    open(config.filename_rouge, 'w')

    model = Transformer(config)
    if torch.cuda.is_available():
        model = model.cuda()

    train(args, config, model)


if __name__ == '__main__':
    main()