import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .highway import Highway


__all__ = [
    "Decoder"
]


class Decoder(nn.Module):

    def __init__(self, feat_size, emb_size, srnn_hidden_size, srnn_num_layers, wrnn_hidden_size, wrnn_num_layers,
                 vocab_size, s_max, w_max, emb_dropout=0., fc_dropout=0.):

        super(Decoder, self).__init__()

        self.feat_size = feat_size
        self.emb_size = emb_size
        self.srnn_hidden_size = srnn_hidden_size
        self.srnn_num_layers = srnn_num_layers

        self.wrnn_hidden_size = wrnn_hidden_size
        self.wrnn_num_layers = wrnn_num_layers
        self.vocab_size = vocab_size
        self.s_max = s_max
        self.w_max = w_max

        self.sentence_rnn = nn.LSTM(input_size=feat_size, hidden_size=srnn_hidden_size, num_layers=srnn_num_layers,
                                    batch_first=True)
        self.cont_stop_layer = nn.Linear(srnn_hidden_size, 2)  # 0 for Continue, 1 for stop
        self.topic_layer = Highway(srnn_hidden_size, emb_size)

        self.embedding_layer = nn.Embedding(vocab_size, emb_size)
        self.emb_dropout_layer = nn.Dropout(p=emb_dropout)

        self.word_rnn = nn.LSTM(input_size=emb_size, hidden_size=wrnn_hidden_size, num_layers=wrnn_num_layers,
                                batch_first=True)

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=fc_dropout),
            nn.Linear(wrnn_hidden_size + emb_size, vocab_size)  # merge structure
        )

    def init_srnn_hidden(self, batch_size, device):

        h0 = torch.zeros(self.srnn_num_layers, batch_size, self.srnn_hidden_size).to(device)
        c0 = torch.zeros(self.srnn_num_layers, batch_size, self.srnn_hidden_size).to(device)

        return h0, c0

    def init_wrnn_hidden(self, batch_size, device):

        h0 = torch.zeros(self.wrnn_num_layers, batch_size, self.wrnn_hidden_size).to(device)
        c0 = torch.zeros(self.wrnn_num_layers, batch_size, self.wrnn_hidden_size).to(device)

        return h0, c0

    def generate_topics(self, visual_feat):
        """Use Sentence RNN to generate s_max topic vectors

        :param visual_feat: (batch_size, feat_size)
        :return: topic_vec (batch_size, s_max, emb_size), cont_stop_unnorm (batch_size, s_max, 2)
        """

        batch_size = visual_feat.shape[0]
        device = visual_feat.device

        srnn_input = visual_feat[:, None, :].expand(batch_size, self.s_max, self.feat_size)
        h0, c0 = self.init_srnn_hidden(batch_size, device)

        srnn_output, _ = self.sentence_rnn(srnn_input, (h0, c0))  # (batch_size, s_max, srnn_hidden_size)

        topic_vec = self.topic_layer(srnn_output)  # (batch_size, s_max, emb_size)
        cont_stop_unnorm = self.cont_stop_layer(srnn_output)  # (batch_size, s_max, 2)

        return topic_vec, cont_stop_unnorm

    def forward(self, visual_feat, encoded_captions, caption_lengths):
        """

        :param visual_feat: (batch_size, enc_size)
        :param encoded_captions: (batch_size, s_max, w_max)
        :param caption_lengths: (batch_size, s_max)
        :return: all_predicts (batch_size, s_max, w_max, vocab_size), con_stop_unnorm (batch_size, s_max, 2)
        """

        batch_size = visual_feat.shape[0]
        device = visual_feat.device

        embeddings = self.embedding_layer(encoded_captions)  # (batch_size, s_max, w_max, embed_size)
        embeddings = self.emb_dropout_layer(embeddings)

        # === Sentence RNN Part ====

        topic_vec, cont_stop_unnorm = self.generate_topics(visual_feat)

        # === Word RNN Part ====

        all_predicts = torch.zeros(batch_size, self.s_max, self.w_max, self.vocab_size).to(device)

        for i in range(self.s_max):

            valid_batch_ind = caption_lengths[:, i] > 0
            valid_batch_size = valid_batch_ind.sum().item()

            if valid_batch_size == 0:
                break

            wrnn_input = embeddings[valid_batch_ind, i]  # (valid_batch_size, w_max, embed_size)
            seq_len = caption_lengths[valid_batch_ind, i]  # (valid_batch_size, )

            wrnn_input_pps = pack_padded_sequence(wrnn_input, lengths=seq_len, batch_first=True, enforce_sorted=False)
            h0, c0 = self.init_wrnn_hidden(valid_batch_size, device)

            wrnn_output_pps, _ = self.word_rnn(wrnn_input_pps, (h0, c0))

            # output (valid_batch_size, w_max, wrnn_hidden_size)
            wrnn_output, _ = pad_packed_sequence(wrnn_output_pps, batch_first=True, total_length=self.w_max)

            # merge structure
            topic = topic_vec[valid_batch_ind, i, None, :].expand(valid_batch_size, self.w_max, self.emb_size)
            fc_input = torch.cat([wrnn_output, topic], dim=-1)  # (valid_batch_size, w_max, wrnn_hidden_size + emb_size)

            predicts = self.fc_layer(fc_input)  # (valid_batch_size, w_max, vocab_size)

            all_predicts[valid_batch_ind, i] = predicts

        return all_predicts, cont_stop_unnorm
