import torch

from model.encoder import Encoder
from model.decoder import Decoder

__all__ = [
    "Captioner"
]


class Captioner(object):

    def __init__(self, encoder, decoder, word2idx, device):
        assert isinstance(encoder, Encoder), "encoder should be an instance of Encoder class"
        assert isinstance(decoder, Decoder), "decoder should be an instance of Decoder class"
        assert encoder.output_size == decoder.feat_size, "encoder and decoder do not match"
        assert decoder.vocab_size == len(word2idx), "decoder and word2idx do not match"
        assert not encoder.training and not decoder.training, 'encoder and decoder should be in eval mode'

        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}

    def to(self, device):

        self.device = device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

    @torch.no_grad()
    def describe_feat(self, feat_input, feat_src='densecap', decode='beam', beam_size=2, verbose=True):

        assert feat_src in {'densecap', 'any'}, "feat_src is either 'densecap' or 'any'."
        assert decode in {'beam', 'greedy', 'greedy_with_penalty'}

        paragraph = list()
        all_cands = list()
        all_scores = list()

        if feat_src == 'densecap':
            assert feat_input.shape == (1, self.encoder.f_max, self.encoder.input_size), "wrong input shape"
            enc_output = self.encoder(feat_input.to(self.device))  # (1, project_size)
        else:  # any tensor with size (1, project_size)
            assert feat_input.shape == (1, self.decoder.feat_size), "wrong input shape"
            enc_output = feat_input.to(self.device)  # (1, project_size)

        topic_vec, cont_stop = self.decoder.generate_topics(enc_output)
        # topic_vec (1, s_max, emb_size)
        # cont_stop (1, s_max, 2)

        cont_stop_flag = torch.log_softmax(cont_stop.squeeze(0), dim=-1).argmax(dim=-1)  # (s_max,)
        cont_stop_flag = torch.where(cont_stop_flag==1)[0]
        if cont_stop_flag.nelement() == 0:
            N = self.decoder.s_max
        else:
            N = torch.min(cont_stop_flag).item() + 1 # 1-for stop

        if verbose:
            print('prepare to generate {} sentences by using {} ...'.format(N, decode))
            if decode=='beam':
                print('using beam size {} ...'.format(beam_size))

        if decode == 'greedy_with_penalty':
            trigrams = dict()
        else:
            trigrams = None

        for i in range(N):
            h0, c0 = self.decoder.init_wrnn_hidden(1, self.device)  # (wrnn_num_layers, 1, wrnn_hidden_size)

            if decode == 'beam':
                best_sent, cands, scores = self.beam_search(topic_vec[:, i, :], beam_size, h0, c0)
            else:
                best_sent = self.greedy_search(topic_vec[:, i, :], h0, c0, trigrams)
                cands = None
                scores = None

            paragraph.append(best_sent)
            all_cands.append(cands)
            all_scores.append(scores)

        return paragraph, all_cands, all_scores

    @torch.no_grad()
    def beam_search(self, topic_vec, beam_size, h0, c0, length_penalty=0.7):
        """
        Ref: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/caption.py
        """
        assert topic_vec.shape[0] == 1, "batch size need to be 1"
        assert isinstance(beam_size, int) and beam_size > 0, "invalid beam size"

        k = beam_size
        vocab_size = self.decoder.vocab_size
        device = self.device

        # We'll treat the problem as having a batch size of k
        k_topic_vec = topic_vec.expand(k, topic_vec.shape[-1])  # (k, emb_size)

        # Tensor to store top k previous words at each step; now they're just <bos>
        k_prev_words = torch.tensor([[self.word2idx['<bos>']]]*k, dtype=torch.long).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <bos>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # start decoding
        h = h0.repeat(1, k, 1)  # (srnn_num_layers, k, wrnn_hidden_size)
        c = c0.repeat(1, k, 1)  # (srnn_num_layers, k, wrnn_hidden_size)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <eos>
        for step in range(self.decoder.w_max):

            embeddings = self.decoder.embedding_layer(k_prev_words)  # (s, 1, emb_size)
            embeddings = self.decoder.emb_dropout_layer(embeddings)

            _, (h, c) = self.decoder.word_rnn(embeddings, (h, c))  # (srnn_num_layers, s, wrnn_hidden_size)

            fc_input = torch.cat((h[-1], k_topic_vec[:k]), 1)  # (s, wrnn_hidden_size + emb_size)

            scores = self.decoder.fc_layer(fc_input)  # (s, vocab_size)
            scores = torch.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <eos>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != self.word2idx['<eos>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[:, prev_word_inds[incomplete_inds], :]
            c = c[:, prev_word_inds[incomplete_inds], :]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if not complete_seqs_scores:
            complete_seqs.extend(seqs.tolist())
            complete_seqs_scores.extend(top_k_scores)

        # Introduce length penalty to scores
        for i in range(len(complete_seqs_scores)):
            sent_length = len(complete_seqs[i])
            complete_seqs_scores[i] *= 1/(sent_length**length_penalty)

        best_idx = complete_seqs_scores.index(max(complete_seqs_scores))
        best_sent = list(self.idx2word[i] for i in complete_seqs[best_idx])

        return best_sent, complete_seqs, complete_seqs_scores

    @torch.no_grad()
    def greedy_search(self, topic_vec, h, c, trigrams=None):

        assert topic_vec.shape[0] == 1, "batch size need to be 1"
        assert trigrams is None or isinstance(trigrams, dict), "trigrams should be None or dict"

        word_idx = torch.tensor([[self.word2idx['<bos>']]], dtype=torch.long).to(self.device)  # (1, 1)

        complete_seqs = list()
        for i in range(self.decoder.w_max):

            embed = self.decoder.embedding_layer(word_idx)  # (1, 1, emb_size)
            embed = self.decoder.emb_dropout_layer(embed)

            _, (h, c) = self.decoder.word_rnn(embed, (h, c))

            fc_input = torch.cat((h[-1], topic_vec), 1)  # (1, wrnn_hidden_size + emb_size)
            log_prob = self.decoder.fc_layer(fc_input).log_softmax(dim=-1)  # (1, vocab_size)

            if isinstance(trigrams, dict):

                # update trigrams
                if i > 2:
                    prev_two = (complete_seqs[i-3], complete_seqs[i-2])
                    if prev_two in trigrams.keys():
                        trigrams[prev_two].append(complete_seqs[i-1])
                    else:
                        trigrams[prev_two] = [complete_seqs[i-1]]

                # update log_prob
                if i > 1:
                    counts = torch.zeros_like(log_prob)  # (1, vocab_size)
                    prev_two = (complete_seqs[i-2], complete_seqs[i-1])
                    for w in trigrams.get(prev_two, []):
                        counts[0][w] += 1

                    # adopt from https://github.com/lukemelas/image-paragraph-captioning
                    alpha = 2.0
                    log_prob = log_prob + (counts * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

            word_idx = log_prob.argmax(dim=-1).unsqueeze(0)  # (1, 1)

            complete_seqs.append(word_idx.item())

            if word_idx.item() == self.word2idx['<eos>']:
                break

        best_sent = list(self.idx2word[i] for i in complete_seqs)

        return best_sent

    def output_cands_with_scores(self, all_cands, all_scores):

        if all_cands[0] is None or all_scores[0] is None:
            return

        assert len(all_cands) == len(all_scores), "Shape doesn't match"

        N = len(all_cands)  # total sentence number

        for i, (cands, scores) in enumerate(zip(all_cands, all_scores)):
            print('sentence {}/{}'.format(i, N))
            for j in sorted(range(len(cands)), key=lambda k: scores[k], reverse=True):
                sent = ' '.join(self.idx2word[idx] for idx in cands[j])
                print('[log_p={:.3f}] {}'.format(scores[j].item(), sent))
