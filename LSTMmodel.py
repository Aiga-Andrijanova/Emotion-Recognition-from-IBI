import json
import numpy as np
import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import torch_optimizer as optim
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from modules.file_utils import FileUtils
from modules.csv_utils_2 import CsvUtils2
from datetime import datetime

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('-sequence_name', default='IBI_2', type=str)
parser.add_argument('-run_name', default='test', type=str)
parser.add_argument('-is_cuda', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-dataset_path', default='data/AllIBIdata.json', type=str)

# Training parameters
parser.add_argument('-epoch_count', default=10, type=int)
parser.add_argument('-learning_rate', default=1e-3, type=float)  # e-3, e-4, e-5
parser.add_argument('-batch_size', default=32, type=int)  # 32, 64, 128

# Model parameters
parser.add_argument('-embedding_size', default=1, type=int)
parser.add_argument('-rnn_layers', default=1, type=int)  # 1 2 3
parser.add_argument('-rnn_dropout', default=0, type=int)  # 0
parser.add_argument('-hidden_size', default=16, type=int)  # 16, 32, 64

parser.add_argument('-early_stopping_patience', default=5, type=int)
parser.add_argument('-early_stopping_param', default='train_loss', type=str)
parser.add_argument('-early_stopping_delta_percent', default=1e-3, type=float)
parser.add_argument('-early_stopping_param_coef', default=1.0, type=float)

args, other_args = parser.parse_known_args()

path_sequence = f'./results/{args.sequence_name}'
args.run_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
path_run = f'./results/{args.sequence_name}/{args.run_name}'
path_artificats = f'./artifacts/{args.sequence_name}/{args.run_name}'
FileUtils.createDir(path_run)
FileUtils.createDir(path_artificats)
FileUtils.writeJSON(f'{path_run}/args.json', args.__dict__)

with open(args.dataset_path) as fp:
    data_json = json.load(fp)
CLASS_COUNT = data_json['class_count']
AROUSAL_WEIGHTS = torch.FloatTensor(data_json['arousal_weights'])
VALENCE_WEIGHTS = torch.FloatTensor(data_json['valence_weights'])
del data_json

CsvUtils2.create_global(path_sequence)
CsvUtils2.create_local(path_sequence, args.run_name)

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
dataset_full = DatasetIBI(args.dataset_path)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full, lengths=[int(len(dataset_full)*0.8), len(dataset_full)-int(len(dataset_full)*0.8)])

torch.seed()  # init random seed


def collate_fn(batch):

    #t_ibi_batch = torch.zeros([len(batch), 25, 1])

    unzipped_batch = zip(*batch)
    unzipped_batch_list = list(unzipped_batch)
    t_lengths_batch = torch.stack(unzipped_batch_list[1]).squeeze(dim=1)
    t_arousal_batch = torch.stack(unzipped_batch_list[2]).squeeze(dim=1)
    t_ibi_batch = torch.stack(unzipped_batch_list[0]).unsqueeze(dim=2)


    indices = torch.argsort(t_lengths_batch, descending=True)

    t_ibi_batch = t_ibi_batch[indices]  # (B, Max_seq, F)
    t_lengths_batch = t_lengths_batch[indices]  # (B, )
    t_arousal_batch = t_arousal_batch[indices].type(torch.LongTensor)  # (B, )

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

        self.linear = torch.nn.Linear(in_features=args.hidden_size, out_features=CLASS_COUNT)

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

model = LSTM(args)
loss_func = torch.nn.CrossEntropyLoss(weight=AROUSAL_WEIGHTS)
optimizer = optim.RAdam(model.parameters(), lr=args.learning_rate)

if args.is_cuda:
    model = model.cuda()
    loss_func = loss_func.cuda()

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = 0

metric_before = {}
metric_mean = {}
early_stopping_patience = 0

state = {
    'epoch': 0,
    'train_loss': -1.0,
    'test_loss': -1.0,
    'best_loss': -1.0,
    'train_acc': -1.0,
    'test_acc': -1.0
}

metrics_epoch = {key: [] for key in metrics.keys()}
metrics_epoch['percent_improvement'] = []
for epoch in range(args.epoch_count):

    percent_improvement = 0

    # Early stopping
    if epoch > 1:
        if metric_before[args.early_stopping_param] != 0:
            if np.isnan(metric_mean[args.early_stopping_param]) or np.isinf(metric_mean[args.early_stopping_param]):
                print('loss isnan break')
                break

            percent_improvement = args.early_stopping_param_coef * (
                        metric_mean[args.early_stopping_param] - metric_before[args.early_stopping_param]) / \
                                  metric_before[args.early_stopping_param]
            if np.isnan(percent_improvement):
                percent_improvement = 0

            if metric_mean[args.early_stopping_param] >= 0:
                if args.early_stopping_delta_percent > percent_improvement:
                    early_stopping_patience += 1
                else:
                    early_stopping_patience = 0
        if early_stopping_patience > args.early_stopping_patience:
            print('early_stopping_patience break')
            break

    for data_loader in [dataloader_train, dataloader_test]:
        metrics_batch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == dataloader_test:
            stage = 'test'

        for x, y, lengths in tqdm(data_loader):

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

            # y_prim = y_prim.cpu()
            # y = y.cpu()

            # values, indices = torch.max(y_prim, dim=1)
            # sum = 0
            # for i in range(len(indices)):
            #     if indices[i] == y[i]:
            #         sum += 1
            # acc = float(sum) / float(indices.__len__())
            y_prim_idxes = torch.argmax(y_prim, dim=1)
            acc = torch.mean((y == y_prim_idxes) * 1.0)

            metrics_batch[f'{stage}_loss'].append(loss.cpu().item())  # Tensor(0.1) => 0.1f
            metrics_batch[f'{stage}_acc'].append(acc.cpu().item())

        metrics_strs = []
        for key in metrics_batch.keys():
            if stage in key:
                value = np.mean(metrics_batch[key])
                metrics_epoch[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')
        metrics_epoch['percent_improvement'].append(percent_improvement)

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    metric_before[args.early_stopping_param] = metrics_epoch[args.early_stopping_param][-1]
    metric_mean[args.early_stopping_param] = np.mean(metrics_epoch[args.early_stopping_param])

    state['train_loss'] = metrics_epoch['train_loss'][-1]
    state['test_loss'] = metrics_epoch['test_loss'][-1]
    state['train_acc'] = metrics_epoch['train_acc'][-1]
    state['test_acc'] = metrics_epoch['test_acc'][-1]
    state['epoch'] = epoch
    if epoch == 0:
        state['best_loss'] = metrics_epoch['test_loss'][-1]
        state['best_acc'] = metrics_epoch['test_acc'][-1]
    if state['test_loss'] < state['best_loss']:
        state['best_loss'] = metrics_epoch['test_loss'][-1]
    if state['test_acc'] < state['best_acc']:
        state['best_acc'] = metrics_epoch['test_acc'][-1]
    state['percent_improvement'] = metrics_epoch['percent_improvement'][-1]

    CsvUtils2.add_hparams(
        path_sequence=path_sequence,
        run_name=args.run_name,
        args_dict=args.__dict__,
        metrics_dict=state,
        global_step=epoch
    )

    plt1 = plt.plot(metrics_epoch['train_loss'], '-b', label='train loss')
    plt2 = plt.plot(metrics_epoch['train_acc'], '--r', label='train acc')
    plt3 = plt.plot(metrics_epoch['test_loss'], '-.c', label='test loss')
    plt4 = plt.plot(metrics_epoch['train_acc'], '-m', label='test acc')

    plts = plt1 + plt2 + plt3 + plt4
    plt.legend(plts, [it.get_label() for it in plts])
    plt.draw()
    plt.pause(0.1)

