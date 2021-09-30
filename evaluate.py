import os
import json
import pickle
import argparse

import torch

from model.encoder import Encoder
from model.decoder import Decoder
from utils.data_loader import CaptionDataset
from captioner import Captioner
from utils.coco import COCO
from utils.DataLoaderPFG import DataLoaderPFG
from pycocoevalcap.eval import COCOEvalCap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COCO_GT_PATHS = {
    'train': './data/coco_gt/para_captions_train.json',
    'test': './data/coco_gt/para_captions_test.json',
    'val': './data/coco_gt/para_captions_val.json'
}

__all__ = [
    "load_args", "load_model", "quantity_evaluate"
]

def load_args(config_path, model_name):

    args = {}

    config_file_path = os.path.join(config_path, model_name, 'config.json')
    print('load configuration file: {} ...'.format(config_file_path))
    with open(config_file_path, 'r') as f:
        args.update(json.load(f))

    word2idx = pickle.load(open(args['word2idx_path'], 'rb'))

    mappings = pickle.load(open(args['mapping_file_path'], 'rb'))

    return args, word2idx, mappings


def load_model(model_checkpoint_path, args):

    encoder = Encoder(input_size=args['input_size'],
                      output_size=args['output_size'],
                      f_max = args['f_max'])

    decoder = Decoder(feat_size=args['feat_size'],
                      emb_size=args['emb_size'],
                      srnn_hidden_size=args['srnn_hidden_size'],
                      srnn_num_layers=args['srnn_num_layers'],
                      wrnn_hidden_size=args['wrnn_hidden_size'],
                      wrnn_num_layers=args['wrnn_num_layers'],
                      vocab_size=args['vocab_size'],
                      s_max=args['s_max'],
                      w_max=args['w_max']-1,
                      emb_dropout=args['emb_dropout'],
                      fc_dropout=args['fc_dropout'])

    print('loading checkpoint from {} ...'.format(model_checkpoint_path))
    checkpoint = torch.load(model_checkpoint_path)
    print('correspond config file: {}'.format(checkpoint['config_path']))
    print('correspond performance on val set: {}'.format(checkpoint['metrics_on_val']))
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    return encoder, decoder


@torch.no_grad()
def quantity_evaluate(encoder, decoder, word2idx, dataset, args, device, decode, beam_size=None, verbose=True,
                      model_config_path=None, mappings=None, is_save_file=False):

    encoder.eval()
    decoder.eval()

    cap = Captioner(encoder, decoder, word2idx, device)

    eval_loader = DataLoaderPFG(CaptionDataset(args['mapping_file_path'], args['visual_features_path'],
                                                args['encoded_paragraphs_path'], dataset),
                                batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    if not mappings:
        mappings = eval_loader.dataset.mappings

    all_candidates = list()
    lengths = list()
    for i, (feats, encoded_caps, cap_lens) in enumerate(eval_loader):

        feats = feats.to(device)

        best_paragraph, _, _ = cap.describe_feat(feats, feat_src='densecap', decode=decode, beam_size=beam_size,
                                                 verbose=False)

        candidate_para = list()
        for sent in best_paragraph:
            candidate_para.extend(w for w in sent if w not in {'<bos>', '<eos>', '<pad>'})
            candidate_para.append('.')

        all_candidates.append(' '.join(candidate_para))
        lengths.append(len(best_paragraph))

        if verbose and i % 500 == 0:
            print('{}/{}'.format(i, len(eval_loader)))
            print('paragraph')
            for sent in best_paragraph:
                print(' '.join(sent))

    coco_format_candidates = list()
    for gid, para in zip(mappings['gid_split_dict'][dataset], all_candidates):
        coco_format_candidates.append({"image_id": mappings['gid2iid'][gid], "caption": para})

    print('Caption Generation Done')
    coco = COCO(COCO_GT_PATHS[dataset])  # load coco format ground truth
    cocoRes = coco.loadRes(coco_format_candidates)  # list or path
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    AVGS = sum(lengths) / len(lengths)
    print('Average sentences per paragraph', AVGS)

    if is_save_file:
        if decode == 'beam':
            cand_path = os.path.join(model_config_path, 'candidate_{}_beam_size_{}.json'.format(dataset, beam_size))
        else:
            cand_path = os.path.join(model_config_path, 'candidate_{}_{}.json'.format(dataset, decode))

        with open(cand_path, 'w') as f:
            json.dump(coco_format_candidates, f)

    metrics = cocoEval.eval.copy()
    metrics['AVGS'] = AVGS
    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./model_params",
                        help="the root directory of all models")
    parser.add_argument("--dataset", type=str, default="test",
                        help="which dataset to evaluate")
    parser.add_argument("--decode", type=str, default="beam",
                        help="decoder type: beam, greedy, greedy_with_penalty")
    parser.add_argument("--beam_size", type=int, default=1,
                        help="beam size of beam search")
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--model_check_point", type=str, help="model checkpoint path")
    args = parser.parse_args()

    config_path = args.config_path
    model_name = args.model_name
    model_check_point = os.path.join(config_path, args.model_check_point)
    dataset = args.dataset
    decode = args.decode
    beam_size = args.beam_size

    config_args, word2idx, mappings = load_args(config_path, model_name)

    print('==================')
    for k, v in config_args.items():
        print('{}     {}'.format(k, v))
    print('==================')

    print('==================')
    encoder, decoder = load_model(model_check_point, config_args)
    print('==================')

    print('decode type {}'.format(decode))

    quantity_evaluate(encoder, decoder, word2idx, dataset, config_args, device, decode, beam_size, True,
                      os.path.join(config_path, model_name), mappings, is_save_file=True)
