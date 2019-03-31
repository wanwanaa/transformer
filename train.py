import argparse
import pickle
from models import *
from utils import *


def valid(epoch, config, model):
    model.eval()
    # data
    valid_loader = data_load(config.filename_trimmed_valid, config.batch_size, False)
    all_loss = 0
    num = 0
    for step, batch in enumerate(valid_loader):
        num += 1
        x, y = batch
        x_pos = torch.arange(1, config.t_len + 1).repeat(x.size(0), 1)
        # print(x_pos.size())
        y_pos = torch.arange(1, config.s_len + 1).repeat(x.size(0), 1)
        # print(y_pos)
        x_mask = x.eq(config.pad)
        # print(x_mask.size())
        x_pos = x_pos.masked_fill(x_mask, 0)
        # y_mask = y.eq(config.pad)
        # y_pos = y_pos.masked_fill(y_mask, 0)
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            x_pos = x_pos.cuda()
            y_pos = y_pos.cuda()
        with torch.no_grad():
            result, loss = model.sample(x, x_pos, y, y_pos)
        all_loss += loss.item()
    print('epoch:', epoch, '|valid_loss: %.4f' % (all_loss / num))
    return all_loss / num


def test(epoch, config, model):
    model.eval()
    # data
    test_loader = data_load(config.filename_trimmed_test, config.batch_size, False)
    all_loss = 0
    num = 0

    # idx2word
    idx2word = pickle.load(open(config.filename_idx2word, 'rb'))
    r = []
    for step, batch in enumerate(test_loader):
        num += 1
        x, y = batch
        x_pos = torch.arange(1, config.t_len+1).repeat(x.size(0), 1)
        # print(x_pos.size())
        y_pos = torch.arange(1, config.s_len+1).repeat(x.size(0), 1)
        # print(y_pos)
        x_mask = x.eq(config.pad)
        # print(x_mask.size())
        x_pos = x_pos.masked_fill(x_mask, 0)
        # y_mask = y.eq(config.pad)
        # y_pos = y_pos.masked_fill(y_mask, 0)
        # print(y_pos)
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            x_pos = x_pos.cuda()
            y_pos = y_pos.cuda()
        with torch.no_grad():
            result, loss = model.sample(x, x_pos, y, y_pos)
        all_loss += loss.item()

        for i in range(result.shape[0]):
            sen = index2sentence(list(result[i]), idx2word)
            r.append(' '.join(sen))
    print('epoch:', epoch, '|test_loss: %.4f' % (all_loss / num))

    # write result
    filename_data = config.filename_data + 'summary_' + str(epoch) + '.txt'
    with open(filename_data, 'w', encoding='utf-8') as f:
        f.write('\n'.join(r))

    # rouge
    score = rouge_score(config.filename_gold, filename_data)

    # write rouge
    write_rouge(config.filename_rouge, score, epoch)

    # print rouge
    print('epoch:', epoch, '|ROUGE-1 f: %.4f' % score['rouge-1']['f'],
          ' p: %.4f' % score['rouge-1']['p'],
          ' r: %.4f' % score['rouge-1']['r'])
    print('epoch:', epoch, '|ROUGE-2 f: %.4f' % score['rouge-2']['f'],
          ' p: %.4f' % score['rouge-2']['p'],
          ' r: %.4f' % score['rouge-2']['r'])
    print('epoch:', epoch, '|ROUGE-L f: %.4f' % score['rouge-l']['f'],
          ' p: %.4f' % score['rouge-l']['p'],
          ' r: %.4f' % score['rouge-l']['r'])

    return score, all_loss / num


def train(args, config, model):
    model.train()
    # optim
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    optim = Optim(optimizer, config)

    # optim = torch.optim.Adam(model.parameters(), lr=0.0003)
    # data
    train_loader = data_load(config.filename_trimmed_train, config.batch_size, True)

    # loss result
    train_loss = []
    valid_loss = []
    test_loss = []
    test_rouge = []

    # # display the result
    # f = open('data/index2word.pkl', 'rb')
    # idx2word = pickle.load(f)

    for e in range(args.epoch):
        model.train()
        all_loss = 0
        num = 0
        for step, batch in enumerate(train_loader):
            num += 1
            x, y = batch
            x_pos = torch.arange(1, config.t_len + 1).repeat(x.size(0), 1)
            y_pos = torch.arange(1, config.s_len + 1).repeat(x.size(0), 1)
            x_mask = x.eq(config.pad)
            x_pos = x_pos.masked_fill(x_mask, 0)
            y_mask = y.eq(config.pad)
            y_pos = y_pos.masked_fill(y_mask, 0)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
                x_pos = x_pos.cuda()
                y_pos = y_pos.cuda()

            out, loss = model(x, x_pos, y, y_pos)

            optim.zero_grad()
            loss.backward()
            optim.updata()
            # optim.step()

            all_loss += loss.item()
            if step % 200 == 0:
                print('epoch:', e, '|step:', step, '|train_loss: %.4f' % loss.item())

            # if step % 2000 == 0:
            #     # display the result
            #     if torch.cuda.is_available():
            #         a = list(y[-1].cpu().numpy())
            #         b = list(torch.argmax(out[-1], dim=1).cpu().numpy())
            #     else:
            #         a = list(y[-1].numpy())
            #         b = list(torch.argmax(out[-1], dim=1).numpy())
            #     a = index2sentence(a, idx2word)
            #     b = index2sentence(b, idx2word)
            #     # display the result
            #     print(''.join(a))
            #     print(''.join(b))

            # if step % 5000 == 0:
            #     test(e, config, model)

            # if step % 2000 == 0:
            #     filename = config.filename_model + 'model_' + str(e) + '_' + str(step) + '.pkl'
            #     save_model(model, filename)
            #     test(e, config, model)

        # train loss
        loss = all_loss / num
        print('epoch:', e, '|train_loss: %.4f' % loss)
        train_loss.append(loss)

        # valid
        loss_v = valid(e, config, model)
        valid_loss.append(loss_v)

        # test
        rouge, loss_t = test(e, config, model)
        test_loss.append(loss_t)
        test_rouge.append(rouge)

        if args.save_model:
            filename = config.filename_model + 'model_' + str(e) + '.pkl'
            save_model(model, filename)


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