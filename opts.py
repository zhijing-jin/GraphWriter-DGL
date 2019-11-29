import os
import torch
import argparse

def fill_config(args):
    args.device = torch.device(args.gpu)
    args.dec_ninp = args.nhid * 3 if args.title else args.nhid * 2
    args.fnames = [args.train_file, args.valid_file, args.test_file]

    download()
    return args

def vocab_config(args, ent_vocab, rel_vocab, text_vocab, ent_text_vocab, title_vocab):
    args.ent_vocab = ent_vocab
    args.rel_vocab = rel_vocab
    args.text_vocab = text_vocab
    args.ent_text_vocab = ent_text_vocab
    args.title_vocab = title_vocab
    return args

def get_args():
    args = argparse.ArgumentParser(description='Graph Writer in DGL')
    args.add_argument('--nhid', default=500, type=int)
    args.add_argument('--nhead', default=4, type=int)
    args.add_argument('--head_dim', default=125, type=int)
    args.add_argument('--weight_decay', default=0.0, type=float)
    args.add_argument('--prop', default=6, type=int)
    args.add_argument('--title', action='store_true')
    args.add_argument('--test', action='store_true')
    args.add_argument('--share_vocab', action='store_true')
    args.add_argument('--batch_size', default=32, type=int)
    args.add_argument('--beam_size', default=4, type=int)
    args.add_argument('--epoch', default=20, type=int)
    args.add_argument('--beam_max_len', default=200, type=int)
    args.add_argument('--enc_lstm_layers', default=2, type=int)
    args.add_argument('--lr', default=1e-1, type=float)
    args.add_argument('--lr_decay', default=1e-8, type=float)
    args.add_argument('--clip', default=1, type=float)
    args.add_argument('--emb_drop', default=0.0, type=float)
    args.add_argument('--attn_drop', default=0.1, type=float)
    args.add_argument('--drop', default=0.1, type=float)
    args.add_argument('--lp', default=1.1, type=float)
    args.add_argument('--graph_enc', default='gtrans', type=str)
    args.add_argument('--train_file', default='data/agenda/train.json',
                      type=str)
    args.add_argument('--valid_file', default='data/agenda/valid.json',
                      type=str)
    args.add_argument('--test_file', default='data/agenda/test.json',
                      type=str)
    args.add_argument('--save_dataset', default='data.pickle', type=str)
    args.add_argument('--save_model', default='saved_model.pt', type=str)

    args.add_argument('--gpu', default=0, type=int)
    args = args.parse_args()
    args = fill_config(args)
    return args

def download():

    if not os.path.isfile('data/agenda/train.json'):
        print('[Info] Downloading AGENDA data...')
        cmd = \
            'curl -O https://raw.githubusercontent.com/rikdz/GraphWriter/master/data/unprocessed.tar.gz; ' \
            'mkdir -p data/agenda/raw; ' \
            'rm data/agenda/raw/* 2>/dev/null; ' \
            'tar -C data/agenda/raw -xvf unprocessed.tar.gz; ' \
            'mv data/agenda/raw/unprocessed.train.json data/agenda/train.json; ' \
            'mv data/agenda/raw/unprocessed.test.json data/agenda/test.json; ' \
            'mv data/agenda/raw/unprocessed.val.json data/agenda/valid.json; ' \
            'rm unprocessed.tar.gz; ' \
            'rm -rf data/agenda/raw/; ' \
            'python data/ablation/full_agenda.py '
        os.system(cmd)


    files = [('eval/detokenizer.perl',
              'https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/detokenizer.perl'),
             ('eval/multi-bleu-detok.perl',
              'https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu-detok.perl')]
    if not os.path.isfile(files[0][0]):
        os.system('mkdir eval')
        for fname, url in files:
            if not os.path.isfile(fname):
                os.system('curl -o {fname} {url}'.format(fname=fname, url=url))

