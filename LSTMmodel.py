import json
import numpy as np
import torch
import torch.utils.data
import argparse
import tqdm
from tqdm import trange
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-is_cuda', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-epoch_count', default=10, type=int)
parser.add_argument('-embedding_size', default=1, type=int)
parser.add_argument('-rnn_layers', default=32, type=int)
parser.add_argument('-rnn_dropout', default=0.3, type=int)
parser.add_argument('-hidden_size', default=16, type=int)
parser.add_argument('-class_count', default=9, type=int)
parser.add_argument('-learning_rate', default=0.001, type=float)
parser.add_argument('-batch_size', default=5, type=int)
args, other_args = parser.parse_known_args()

class DatasetIBI(torch.utils.data.Dataset):
    def __init__(
            self,
            path_data_json: str
    ):
        super().__init__()

        with open(path_data_json) as fp:
            data_json = json.load(fp)

        path_data_mmap = path_data_json.replace('.json', '.mmap')
        data_mmap = np.memmap(
            path_data_mmap,
            mode='r',
            dtype=np.float16,
            shape=tuple(data_json['shape'])
        )

        self.data = []
        for idx_sample in range(data_json['shape'][0]):
            self.data.append([
                data_mmap[idx_sample],  # (Max_seq_len, )
                data_json['lengths'][idx_sample],
                data_json['arousal'][idx_sample],
                data_json['valence'][idx_sample]]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ibi, length, arousal, valance = self.data[idx]
        t_ibi = torch.FloatTensor(ibi)
        t_length = torch.IntTensor([length])
        t_arousal = torch.IntTensor([arousal])
        # t_valance = torch.IntTensor([valance])
        return t_ibi, t_length, t_arousal


torch.manual_seed(0)  # lock seed
dataset_full = DatasetIBI('Data/AllIBIdata.json')
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full, lengths=[int(len(dataset_full)*0.8), len(dataset_full)-int(len(dataset_full)*0.8)])

torch.seed()  # init random seed


def collate_fn(batch):

    t_ibi_batch = torch.zeros([len(batch), 25, 1])

    unzipped_batch = zip(*batch)
    unzipped_batch_list = list(unzipped_batch)
    t_lengths_batch = torch.stack(unzipped_batch_list[1]).squeeze(dim=1)
    t_arousal_batch = torch.stack(unzipped_batch_list[2]).squeeze(dim=1)

    i = 0
    for t_ibi, t_lengths, t_arousal in batch:
        t_ibi_batch[i, :, :] = t_ibi.unsqueeze(dim=1)

        i = i + 1
    t_ibi_batch.squeeze(dim=1)

    indices = torch.argsort(t_lengths_batch, descending=True)

    t_ibi_batch = t_ibi_batch[indices]  # (B, Max_seq, F)
    t_lengths_batch = t_lengths_batch[indices]  # (B, )
    t_arousal_batch = t_arousal_batch[indices].type(torch.LongTensor)  # (B, )
    t_arousal_batch = torch.add(t_arousal_batch, -1)  # [1;9] -> [0;8]

    return t_ibi_batch, t_arousal_batch, t_lengths_batch


dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=args.batch_size,
    collate_fn=collate_fn,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=args.batch_size,
    collate_fn=collate_fn,
    shuffle=False
)


class LSTM(torch.nn.Module):
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

        temp_batch_size = unpacked_rnn_out.size()[0]
        temp_max_seq_len = unpacked_rnn_out.size()[1]
        temp_hidden_size = unpacked_rnn_out.size()[2]

        h = torch.zeros((temp_batch_size, temp_hidden_size)).cuda()  # (B, F)
        for sample in range(unpacked_rnn_out.size()[0]):
            h[sample] = torch.mean(unpacked_rnn_out[sample, :unpacked_rnn_out_lenghts[sample]], axis=0)

        out = self.linear(h)
        #out = F.softmax(out, dim=1)
        return out

model = LSTM(args)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

if args.is_cuda:
    model = model.cuda()
    loss_func = loss_func.cuda()

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss'
    ]:
        metrics[f'{stage}_{metric}'] = 0

for epoch in range(args.epoch_count):

    for data_loader in [dataloader_train, dataloader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == dataloader_test:
            stage = 'test'

        for x, y, lengths in tqdm(data_loader):

            model.zero_grad()

            padded_packed = pack_padded_sequence(x, lengths, batch_first=True)

            if args.is_cuda:
                y = y.cuda()
                padded_packed = padded_packed.cuda()

            y_prim = model.forward(padded_packed)

            loss = loss_func.forward(y_prim, y)

            if data_loader == dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # move all data back to cpu
            # loss = loss.cpu()
            # y_prim = y_prim.cpu()
            # y = y.cpu()
            # x = x.cpu()

            metrics_epoch[f'{stage}_loss'].append(loss.item())  # Tensor(0.1) => 0.1f

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key] = value
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')
