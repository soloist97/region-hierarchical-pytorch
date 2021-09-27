import os
import json
import h5py
import pickle
import string
from collections import Counter, OrderedDict

import spacy
import numpy as np
from tqdm import tqdm


class PreProcessor(object):

    def __init__(self, para_file_path, output_file_path, min_word_per_sent):

        print('[INFO]: loading spacy model...')
        self.parser = spacy.load('en_core_web_lg')
        with open(para_file_path, 'r') as f:
            self.raw_data = json.load(f)
        self.output_file_path = output_file_path
        self.min_word_per_sent = min_word_per_sent

    def split_to_words(self, sent_str):

        # replace special tokens
        replacements = {
            u'½': u'half',
            u'—': u'-',
            u'™': u'',
            u'¢': u'cent',
            u'ç': u'c',
            u'û': u'u',
            u'é': u'e',
            u'°': u' degree',
            u'è': u'e',
            u'…': u'',
        }

        for k, v in replacements.items():
            sent_str = sent_str.replace(k, v)

        # tokenizer
        words = list()

        for tok in self.parser.tokenizer(sent_str):
            if ' ' not in tok.text and not all(c in string.punctuation for c in tok.text):
                words.append(tok.text.lower())

        return words

    def split_to_sentences(self, para_str):

        # dirty handy rules
        replacements = OrderedDict({
            '.T': '. T',
            '.th': '. th',
            '.A': '. A',
            '.a': '. a',
            't.v.': 'tv',
            '...': '. ',
            '..': '. ',
        })

        for k, v in replacements.items():
            para_str = para_str.replace(k, v)

        # split to sentences
        words = list()
        for tok in self.parser(para_str.strip()):
            if ' ' not in tok.text:
                words.append(tok.text)

        sentences = list(s for s in (' '.join(words)+' ').split(' . ') if len(s) > 0)

        return sentences

    def process(self, verbose=True):

        output_dict = list()

        for r_data in tqdm(self.raw_data, disable=not verbose):

            raw_sentences = self.split_to_sentences(r_data['paragraph'])

            sentences = list()
            for sent in raw_sentences:

                words = self.split_to_words(sent)

                if len(words) < self.min_word_per_sent:
                    tqdm.write('[Image id]: '+str(r_data['image_id']))
                    tqdm.write('[Problematic phrase]: ' + ' '.join(words))
                    tqdm.write('\n'.join(raw_sentences)+'\n')
                    tqdm.write('-'*20)
                    continue

                sentences.append(' '.join(words))

            output_dict.append({
                'url': r_data['url'],
                'image_id': r_data['image_id'],
                'paragraph': r_data['paragraph'],
                'sentences': sentences
            })

        with open(self.output_file_path, 'w') as f:
            json.dump(output_dict, f, indent=4)


def map_features(mapping_file_path, features_path_dict, feat_file_path, f_max=50, f_size=4096, verbose=True):

    if verbose:
        print('[INFO]: load mapping file from', mapping_file_path)
    with open(mapping_file_path, 'rb') as f:
        mappings = pickle.load(f)

    with h5py.File(feat_file_path, 'w') as h:

        # === record important meta settings ====
        h.attrs['mapping_file_path'] = mapping_file_path

        feats = h.create_dataset('feats', (len(mappings['gid2iid']), f_max, f_size), dtype=np.float)
        boxes = h.create_dataset('boxes', (len(mappings['gid2iid']), f_max, 4), dtype=np.float)

        for dataset_name, paths in features_path_dict.items():
            if verbose:
                print('[INFO]: mapping {} ...'.format(dataset_name))
                print('[INFO]: loading iid path from {} ...'.format(paths['iid_path']))
                print('[INFO]: mapping feat based on {} ...'.format(paths['feat_path']))

            feat_iids = list()
            with open(paths['iid_path'], 'r') as f:
                for path in f.readlines():
                    file_name = os.path.split(path.split()[0])[1]
                    feat_iids.append(int(os.path.splitext(file_name)[0]))

            with h5py.File(paths['feat_path'], 'r') as dh:
                for i, iid in tqdm(enumerate(feat_iids), disable=not verbose, total=len(feat_iids)):
                    gid = mappings['iid2gid'][iid]

                    feats[gid] = dh['feats'][i]
                    boxes[gid] = dh['boxes'][i]

        if verbose:
            print('[INFO]: features saved to', feat_file_path)


def create_encoded_paragraph(preprocessed_file_path, mapping_file_path, vocab_file_path, encoded_file_path,
                             max_sent_per_para, max_word_per_sent, verbose=True):
    if verbose:
        print('[INFO]: load preprocessed file from', preprocessed_file_path)
        print('[INFO]: load mapping file from', mapping_file_path)
        print('[INFO]: load vocab file from', vocab_file_path)
    with open(preprocessed_file_path, 'r') as f:
        preprocessed_data = json.load(f)
    with open(mapping_file_path, 'rb') as f:
        mappings = pickle.load(f)
    with open(vocab_file_path, 'rb') as f:
        word2idx = pickle.load(f)

    with h5py.File(encoded_file_path, 'w') as h:

        # === record important meta settings ====
        h.attrs['s_max'] = max_sent_per_para
        h.attrs['w_max'] = max_word_per_sent
        h.attrs['preprocessed_file_path'] = preprocessed_file_path
        h.attrs['mapping_file_path'] = mapping_file_path
        h.attrs['vocab_file_path'] = vocab_file_path

        paragraphs = h.create_dataset('encoded_paragraph', (len(mappings['gid2iid']), max_sent_per_para,
                                                            max_word_per_sent), dtype=np.int64)
        lengths = h.create_dataset('length', (len(mappings['gid2iid']), max_sent_per_para), dtype=np.int64)

        for gid in tqdm(range(len(mappings['gid2pid'])), disable= not verbose):
            data = preprocessed_data[mappings['gid2pid'][gid]]

            para = np.ones([max_sent_per_para, max_word_per_sent], dtype=np.int64) * word2idx['<pad>']
            len_ = np.zeros([max_sent_per_para,], dtype=np.int64)

            for i in range(min(len(data['sentences']), max_sent_per_para)):
                words = data['sentences'][i].split()
                assert len(words) >= 3, "invalid length of sentence"
                words = ['<bos>'] + words[:max_word_per_sent-2] + ['<eos>']

                len_[i] = min(len(words), max_word_per_sent)
                para[i, :len_[i]] = np.array([word2idx.get(w, word2idx['<unk>']) for w in words])

            paragraphs[gid] = para
            lengths[gid] = len_

            if verbose:
                print('[INFO]: encoded paragraph saved to', encoded_file_path)


def create_vocabulary(preprocessed_file_path, mapping_file_path, vocab_file_path, min_freq, verbose=True):

    if verbose:
        print('[INFO]: load preprocessed file from', preprocessed_file_path)
        print('[INFO]: load mapping file from', mapping_file_path)

    with open(preprocessed_file_path, 'r') as f:
        preprocessed_data = json.load(f)
    with open(mapping_file_path, 'rb') as f:
        mappings = pickle.load(f)

    word_counter = Counter()
    for gid in tqdm(mappings['gid_split_dict']['train'], disable=not verbose):
        pid = mappings['gid2pid'][gid]
        for sent in preprocessed_data[pid]['sentences']:
            word_counter.update(sent.split())

    vocab = set()
    for word, count in word_counter.items():
        if count >= min_freq:
            vocab.add(word)

    if verbose:
        print('[INFO]: Keeping {} / {} words'.format(len(vocab), len(word_counter)))

    vocab = list(vocab)
    vocab = sorted(vocab, key=lambda token: word_counter[token], reverse=True)

    vocab = ['<pad>', '<bos>', '<eos>', '<unk>'] + vocab
    if verbose:
        print('[INFO]: Adding special <pad> <bos> <eos> <unk> token.')

    word2idx = {w: i for i, w in enumerate(vocab)}

    with open(vocab_file_path, 'wb') as f:
        pickle.dump(word2idx, f)
    if verbose:
        print('[INFO]: save word2idx file to', vocab_file_path)


def create_mapping(preprocessed_file_path, split_path_dict, mapping_file_path, verbose=True):

    assert {'train', 'test', 'val'} == set(split_path_dict.keys())

    split_iid_dict = dict()
    for dataset_name, split_file_path in split_path_dict.items():
        with open(split_file_path, 'r') as f:
            split_iid_dict[dataset_name] = json.load(f)

    gid = 0
    iid2gid = dict()  # image_id to global_id
    gid_split_dict = dict()
    for dn in ['train', 'val', 'test']:
        gid_split_dict[dn] = list()
        for iid in split_iid_dict[dn]:
            gid_split_dict[dn].append(gid)
            iid2gid[iid] = gid
            gid += 1
    gid2iid = {gid: iid for iid, gid in iid2gid.items()}

    with open(preprocessed_file_path, 'r') as f:
        preprocessed_data = json.load(f)

    gid2pid = dict()  # global_id to paragraph_id
    for pid, data in enumerate(preprocessed_data):
        gid = iid2gid[data['image_id']]
        if gid not in gid2pid.keys():
            gid2pid[gid] = pid
        else:
            if verbose:
                print('[INFO]: {} have different ground truth paragraphs'.format(data['image_id']))
            if len(preprocessed_data[pid]['sentences']) > len(preprocessed_data[gid2pid[gid]]['sentences']):
                if verbose:
                    print('[INFO]: pid-{} is longer than pid-{}, choose pid-{}\n'.format(pid, gid2pid[gid], pid))
                gid2pid[gid] = pid
            else:
                if verbose:
                    print('[INFO]: pid-{} is smaller than or equal to pid-{}, keep pid-{}\n'.format(pid, gid2pid[gid],
                                                                                                    gid2pid[gid]))
    pid2gid = {pid: gid for gid, pid in gid2pid.items()}

    mappings = {
        'gid2iid': gid2iid,
        'iid2gid': iid2gid,
        'gid2pid': gid2pid,
        'pid2gid': pid2gid,
        'gid_split_dict': gid_split_dict,
    }

    if verbose:
        print('[INFO]: mappings save to {}'.format(mapping_file_path))
    with open(mapping_file_path, 'wb') as f:
        pickle.dump(mappings, f)


if __name__ == '__main__':

    # === step 2: create preprocessed paragraph file ====
    # s_min_list = [3, 4, 5, 6]
    
    # for s_min in s_min_list:
    #     pp = PreProcessor(para_file_path='./data/stanfordParagraph/paragraphs/paragraphs_v1.json',
    #                       output_file_path='./data/cleaned/preprocessed_paragraph_s_min_{}.json'.format(s_min),
    #                       min_word_per_sent=s_min)
    
    #     pp.process(verbose=True)

    # === step 3: create mappings ====
    # s_min = 3
    # pfp = './data/cleaned/preprocessed_paragraph_s_min_{}.json'.format(s_min)
    # spd = {
    #     'train': './data/stanfordParagraph/paragraphs/train_split.json',
    #     'test': './data/stanfordParagraph/paragraphs/test_split.json',
    #     'val': './data/stanfordParagraph/paragraphs/val_split.json'
    # }
    # mfp = './data/cleaned/mappings.pkl'.format(s_min)
    #
    # create_mapping(pfp, spd, mfp, verbose=True)

    # === step 4: create vocab file ====
    # s_min = 3
    # w_min = 2
    # mfp = './data/cleaned/mappings.pkl'
    #
    # pfp = './data/cleaned/preprocessed_paragraph_s_min_{}.json'.format(s_min)
    # vfp = './data/cleaned/word2idx_s_min_{}_w_min_{}.pkl'.format(s_min, w_min)
    #
    # create_vocabulary(pfp, mfp, vfp, w_min, verbose=True)

    # === step 5: encode paragraphs ====
    # s_max = 6
    # w_max = 33
    #
    # s_min = 3
    # w_min = 2
    #
    # pfp = './data/cleaned/preprocessed_paragraph_s_min_{}.json'.format(s_min)
    # mfp = './data/cleaned/mappings.pkl'
    # vfp = './data/cleaned/word2idx_s_min_{}_w_min_{}.pkl'.format(s_min, w_min)
    # efp = './data/cleaned/encoded_paragraphs_s_{}_{}_w_{}_{}.h5'.format(s_min, s_max, w_min, w_max)
    #
    # create_encoded_paragraph(pfp, mfp, vfp, efp, s_max, w_max, verbose=True)

    # === step 6: map densecap features ====
    # Base file created follow: https://github.com/chenxinpeng/im2p
    mfp = './data/cleaned/mappings.pkl'
    fpd = {
        'train':{
            'feat_path': './data/densecap/im2p_train_output.h5',
            'iid_path': './data/densecap/imgs_train_path.txt'
        },
        'test': {
            'feat_path': './data/densecap/im2p_test_output.h5',
            'iid_path': './data/densecap/imgs_test_path.txt'
        },
        'val': {
            'feat_path': './data/densecap/im2p_val_output.h5',
            'iid_path': './data/densecap/imgs_val_path.txt'
        }
    }
    ffp = './data/cleaned/densecap_image_features_f_max_50_f_size_4096.h5'

    map_features(mfp, fpd, ffp, f_max=50, f_size=4096, verbose=True)
