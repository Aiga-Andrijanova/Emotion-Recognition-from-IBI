import torch
import torch.utils.data
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.rnn = torch.nn.LSTM(
            input_size=1,
            hidden_size=args.hidden_size,
            num_layers=args.rnn_layers,
            dropout=args.rnn_dropout,
            batch_first=True
        )

        self.linear = torch.nn.Linear(in_features=args.hidden_size, out_features=args.class_count)

    def forward(self, x: PackedSequence):

        packed_rnn_out_data, (_, _) = self.rnn.forward(x)
        unpacked_rnn_out, unpacked_rnn_out_lenghts = pad_packed_sequence(packed_rnn_out_data, batch_first=True)

        batch_size = unpacked_rnn_out.size()[0]
        max_seq_len = unpacked_rnn_out.size()[1]
        hidden_size = unpacked_rnn_out.size()[2]

        h = torch.zeros((batch_size, hidden_size)).cuda()
        # (B, F)
        for idx_sample in range(batch_size):
            len_sample = unpacked_rnn_out_lenghts[idx_sample]
            h_sample = unpacked_rnn_out[idx_sample, :len_sample]
            h[idx_sample] = torch.mean(h_sample, axis=0)

        out = self.linear(h)
        return out