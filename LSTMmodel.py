import json
import numpy as np
import torch
import torch.utils.data
import argparse
import tqdm
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-embedding_size', default=10, type=int)
parser.add_argument('-rnn_layers', default=32, type=int)
parser.add_argument('-rnn_dropout', default=0.3, type=int)
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
        t_valance = torch.IntTensor([valance])
        return t_ibi, t_length, t_arousal, t_valance


torch.manual_seed(0)  # lock seed
dataset_full = DatasetIBI('Data/AllIBIdata.json')
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full, lengths=[int(len(dataset_full)*0.8), len(dataset_full)-int(len(dataset_full)*0.8)])

torch.seed()  # init random seed


def collate_fn(batch):

    t_ibi = batch[:][0]
    t_lengths = batch[:][1]
    t_arousal = batch[:][2]
    indices = torch.argsort(t_lengths, descending=True)
    t_ibi = [t_ibi[i] for i in indices]
    # t_ibi = t_ibi[None, :, :]
    # t_lengths = t_lengths[None, :]
    # t_arousal = t_arousal[None, :]
    # t_valance = t_valance[None, :]
    # return t_ibi, t_lengths, t_arousal, t_valance
    return t_ibi, t_arousal, t_lengths  #t_ibi un t_arousal ir tuple, kas sastāv no vairākiem tensoriem


dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=10,
    collate_fn=collate_fn,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=10,
    collate_fn=collate_fn,
    shuffle=False
)


class LSTM(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.rnn = torch.nn.LSTM(
            input_size=args.embedding_size,
            hidden_size=args.embedding_size,
            num_layers=args.rnn_layers,
            dropout=args.rnn_dropout,
            batch_first=True
        )

    def forward(self, x: PackedSequence, hx=None):

        if isinstance(x, PackedSequence):
            emb = PackedSequence(
                x.data,
                x.batch_sizes,
                x.sorted_indices,
                x.unsorted_indices
            )
        out_seq, (hn, cn) = self.rnn.forward(x, hx)
        ret = torch.matmul(out_seq.data, self.embedding.weight.t())
        if isinstance(x, PackedSequence):
            prob = torch.softmax(ret, dim=1)
            prob = PackedSequence(
                prob,
                x.batch_sizes,
                x.sorted_indices,
                x.unsorted_indices
            )
        else:
            prob = torch.softmax(ret, dim=2)

        return prob, (hn, cn)

model = LSTM(args)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss'
    ]:
        metrics[f'{stage}_{metric}'] = 0

for epoch in range(10):

    for data_loader in [dataloader_train, dataloader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == dataloader_test:
            stage = 'test'

        for x, y, lengths in data_loader:

            model.zero_grad()

            padded_packed = pack_padded_sequence(x, lengths)
            y_prim = model.forward(x)
            loss = loss_func.forward(y_prim, y)

            if data_loader == dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            else:
                hidden = model.init_hidden(t_ibi.size(0))
                y_prim, hidden = model.forward(t_ibi, hidden)
                for _ in range(5):
                    input = y_prim[:, -1].unsqueeze(-1)
                    y_prim_step, hidden = model.forward(input, hidden)
                    y_prim = torch.cat([y_prim, y_prim_step], dim=1)

            # move all data back to cpu

            metrics_epoch[f'{stage}_loss'].append(loss.item()) # Tensor(0.1) => 0.1f

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key] = value
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

# Padodot uz LSTM bus vajadzigs PackedSequence
#  (https://www.notion.so/evalds/PackedSequence-RNN-5922eaeea48644a2b96fe5e13ef1a185)