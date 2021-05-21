import torch
import torch.utils.data
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as FF


class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.ff = torch.nn.Linear(
            in_features=1,
            out_features=args.hidden_size,
        )

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)  # aka He normal
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        self.conv4 = torch.nn.Conv2d(in_channels=256, out_channels=96, kernel_size=3, stride=1, padding=1)
        torch.nn.init.kaiming_normal_(self.conv4.weight)


        self.linear = torch.nn.Linear(in_features=96, out_features=args.class_count)

    def forward(self, x, lengths):
        # x (B, max_seq_len_, 1)
        x_fourier = torch.stft(x.squeeze(dim=-1), n_fft=20, hop_length=1, win_length=20, return_complex=True)

        # x_prim = self.ff.forward(x)

        conv_out = self.conv1(x_fourier.unsqueeze(dim=1))
        conv_out = FF.leaky_relu(conv_out)
        conv_out = self.conv2(conv_out)
        conv_out = FF.leaky_relu(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = FF.leaky_relu(conv_out)
        conv_out = self.conv4(conv_out)
        conv_out = FF.leaky_relu(conv_out)
        conv_out = FF.avg_pool1d(conv_out, kernel_size=conv_out.shape[-1])  # output => (B, F, 1)

        conv_out = self.linear(conv_out)
        return conv_out