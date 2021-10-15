import torch
import torch.utils.data
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as FF


class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.linear = torch.nn.Linear(in_features=64, out_features=args.class_count)

    def forward(self, x, lengths):
        x_fourier = torch.stft(x.squeeze(dim=-1), n_fft=20, hop_length=1, win_length=20, return_complex=False)
        x_fourier = x_fourier.permute(0, 3, 1, 2)

        conv_out = self.conv1(x_fourier)
        conv_out = FF.leaky_relu(conv_out)
        conv_out = self.conv2(conv_out)
        conv_out = FF.leaky_relu(conv_out)
        conv_out = self.conv3(conv_out)
        conv_out = FF.leaky_relu(conv_out)
        conv_out = self.conv4(conv_out)
        conv_out = FF.leaky_relu(conv_out)
        conv_out = self.conv5(conv_out)
        conv_out = FF.leaky_relu(conv_out)
        conv_out = FF.adaptive_avg_pool2d(conv_out, output_size=(1, 1))

        conv_out = self.linear(conv_out.squeeze(dim=-1).squeeze(dim=-1))
        return conv_out
