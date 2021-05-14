import torch
import torch.utils.data
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.ff = torch.nn.Linear(
            in_features=1,
            out_features=args.hidden_size,
        )

        self.rnn = torch.nn.LSTM(
            input_size=args.hidden_size,
            hidden_size=args.hidden_size,
            num_layers=args.rnn_layers,
            dropout=args.rnn_dropout,
            batch_first=True
        )

        indices = []
        for i in range(args.hidden_size):
            indices.append(1 + i * 4)

        self.rnn.bias_hh_l0.data[indices] = torch.ones(args.hidden_size)
        self.rnn.bias_ih_l0.data[indices] = torch.ones(args.hidden_size)
        if args.rnn_layers == 2 or args.rnn_layers == 3:
            self.rnn.bias_hh_l1.data[indices] = torch.ones(args.hidden_size)
            self.rnn.bias_ih_l1.data[indices] = torch.ones(args.hidden_size)
        if args.rnn_layers == 3:
            self.rnn.bias_hh_l2.data[indices] = torch.ones(args.hidden_size)
            self.rnn.bias_ih_l2.data[indices] = torch.ones(args.hidden_size)

        self.linear = torch.nn.Linear(in_features=args.hidden_size * 3, out_features=args.class_count)

    def forward(self, x, lengths):
        x_packed = pack_padded_sequence(x, lengths, batch_first=True)
        x_prim = self.ff.forward(x_packed.data)

        x_prim_seq = PackedSequence(
            data=x_prim,
            batch_sizes=x_packed.batch_sizes
        )

        packed_rnn_out_data, (_, _) = self.rnn.forward(x_prim_seq)
        unpacked_rnn_out, unpacked_rnn_out_lengths = pad_packed_sequence(packed_rnn_out_data, batch_first=True)

        batch_size = unpacked_rnn_out.size(0)
        max_seq_len = unpacked_rnn_out.size(1)
        hidden_size = unpacked_rnn_out.size(2)

        h = torch.zeros((batch_size, hidden_size * 3)).cuda()
        # (B, F)
        for idx_sample in range(batch_size):
            len_sample = unpacked_rnn_out_lengths[idx_sample]
            h_sample = unpacked_rnn_out[idx_sample, :len_sample]
            h[idx_sample, :hidden_size] = torch.mean(h_sample, axis=0)
            h[idx_sample, hidden_size:hidden_size * 2] = torch.max(h_sample, axis=0).values
            h[idx_sample, hidden_size * 2:hidden_size * 3] = h_sample[-1]

        out = self.linear(h)
        return out