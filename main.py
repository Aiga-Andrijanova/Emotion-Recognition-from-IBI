import json
import numpy as np
import numpy.ma
import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import torch_optimizer as optim
import matplotlib.pyplot as plt
from modules.file_utils import FileUtils
from modules.csv_utils_2 import CsvUtils2
from datetime import datetime
import time
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-model', default='LSTM_V2', type=str)

parser.add_argument('-sequence_name', default='Fourier_2C_grid_search', type=str)
parser.add_argument('-run_name', default='run_4', type=str)
parser.add_argument('-device', default='cpu', type=str)
parser.add_argument('-dataset_path', default='./data_processed_full/AMIGOS_IBI_30sec_2C_AllVideos_minmax_byperson.json', type=str)
# ./data/DREAMER_IBI_30sec_byperson.json
# ./data/AMIGOS_IBI_30sec_byseq.json

# Training parameters
parser.add_argument('-epoch_count', default=3, type=int)
parser.add_argument('-learning_rate', default=1e-4, type=float)
parser.add_argument('-batch_size', default=4, type=int)

# Model parameters
parser.add_argument('-embedding_size', default=1, type=int)
parser.add_argument('-rnn_layers', default=1, type=int)
parser.add_argument('-rnn_dropout', default=0.25, type=float)
parser.add_argument('-hidden_size', default=8, type=int)

parser.add_argument('-kernel_size', default=1, type=int)
parser.add_argument('-conv_dropout', default=0.25, type=float)

parser.add_argument('-early_stopping_patience', default=5, type=int)
parser.add_argument('-early_stopping_param', default='train_loss', type=str)
parser.add_argument('-early_stopping_delta_percent', default=1e-3, type=float)
parser.add_argument('-early_stopping_param_coef', default=1.0, type=float)

args, other_args = parser.parse_known_args()

def validate(model, dataloader_validation, args, loss_func, state, CsvUtils2):
    stage = 'validation'
    model.eval()

    metrics = {}
    for stage in ['validation']:
        for metric in [
            'loss',
            'acc',
            'F1'
        ]:
            metrics[f'{stage}_{metric}'] = 0
    metrics_epoch = {key: [] for key in metrics.keys()}

    for data_loader in [dataloader_validation]:

        metrics_batch = {key: [] for key in metrics.keys()}
        conf_matrix = numpy.zeros((args.class_count, args.class_count))

        for x, y, lengths in data_loader:

            y = y.to(args.device)
            x = x.to(args.device)

            y_prim = model.forward(x, lengths)

            loss = loss_func.forward(y_prim, y)

            y_prim_idxes = torch.argmax(y_prim, dim=1)
            acc = torch.mean((y == y_prim_idxes) * 1.0)

            F1_score = f1_score(y.to('cpu'), y_prim_idxes.to('cpu'), average='micro', zero_division=0)

            metrics_batch[f'{stage}_loss'].append(loss.cpu().item())  # Tensor(0.1) => 0.1f
            metrics_batch[f'{stage}_acc'].append(acc.cpu().item())
            metrics_batch[f'{stage}_F1'].append(F1_score)

            for idx, y_prim_idx in enumerate(y_prim_idxes):
                y_idx = y[idx]
                conf_matrix[y_idx.item(), y_prim_idx.item()] += 1

        metrics_strs = []
        for key in metrics_batch.keys():
            if stage in key:
                value = np.mean(metrics_batch[key])
                metrics_epoch[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'Validation: {" ".join(metrics_strs)}')

        state['validation_loss'] = metrics_epoch['validation_loss'][-1]
        state['validation_acc'] = metrics_epoch['validation_acc'][-1]
        state['validation_F1'] = metrics_epoch['validation_F1'][-1]

        CsvUtils2.add_hparams(
            path_sequence=path_sequence,
            run_name=args.run_name,
            args_dict=args.__dict__,
            metrics_dict=state,
            global_step=args.epoch_count + 1
        )


path_sequence = f'./results/{args.sequence_name}'
args.run_name += ('-' + datetime.utcnow().strftime(f'%y-%m-%d--%H-%M-%S'))
path_run = f'./results/{args.sequence_name}/{args.run_name}'
path_artifacts = f'./artifacts/{args.sequence_name}/{args.run_name}'
FileUtils.createDir(path_run)
FileUtils.createDir(path_artifacts)
FileUtils.writeJSON(f'{path_run}/args.json', args.__dict__)

with open(args.dataset_path) as fp:
    data_json = json.load(fp)

parser.add_argument('-class_count', default=data_json['class_count'], type=int)
parser.add_argument('-max_seq_len', default=data_json['shape'][1], type=int)
args, other_args = parser.parse_known_args()

AROUSAL_WEIGHTS = torch.FloatTensor(data_json['valence_weights'])
VALENCE_WEIGHTS = torch.FloatTensor(data_json['valence_weights'])
del data_json

CsvUtils2.create_global(path_sequence)
CsvUtils2.create_local(path_sequence, args.run_name)

class DatasetIBI(torch.utils.data.Dataset):
    def __init__(self, path_data_json: str, person_split):
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
            if data_json['person_id'][idx_sample] in person_split:
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
        t_ibi = torch.FloatTensor(ibi.copy())
        t_length = torch.IntTensor([length])
        # t_arousal = torch.IntTensor([arousal])
        t_valence = torch.IntTensor([valance])
        return t_ibi, t_length, t_valence


if 'AMIGOS' in args.dataset_path:
    test_person_split = [2, 4, 6, 7, 19, 23, 25, 27, 34, 36, 37, 40]
    train_person_split = [1, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20,
                          21, 22, 24, 26, 28, 29, 31, 31, 32, 33, 35, 38, 39]
else:
    test_person_split = [1, 4, 5, 6, 7, 8, 11]
    train_person_split = [2, 3, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                          22, 23]

torch.manual_seed(0)  # lock seed
dataset_train = DatasetIBI(args.dataset_path, person_split=train_person_split)
dataset_nontrain = DatasetIBI(args.dataset_path, person_split=test_person_split)

test_data_len = int(len(dataset_nontrain) * 0.7)  # ~21% of all data
valid_data_len = int(len(dataset_nontrain) - test_data_len)  # ~9% of all data

dataset_test, dataset_validation = torch.utils.data.random_split(
    dataset_nontrain, lengths=[test_data_len, valid_data_len])

torch.manual_seed(int(time.time()))  # init random seed


def collate_fn(batch):

    unzipped_batch = zip(*batch)
    unzipped_batch_list = list(unzipped_batch)
    t_lengths_batch = torch.stack(unzipped_batch_list[1]).squeeze(dim=1)
    t_valence_batch = torch.stack(unzipped_batch_list[2]).squeeze(dim=1)
    t_ibi_batch = torch.stack(unzipped_batch_list[0]).unsqueeze(dim=2)

    indices = torch.argsort(t_lengths_batch, descending=True)

    t_ibi_batch = t_ibi_batch[indices]  # (B, Max_seq, F)
    t_lengths_batch = t_lengths_batch[indices]  # (B, )
    t_valence_batch = t_valence_batch[indices].type(torch.LongTensor)  # (B, )

    return t_ibi_batch, t_valence_batch, t_lengths_batch


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

dataloader_validation = torch.utils.data.DataLoader(
    dataset_validation,
    batch_size=args.batch_size,
    collate_fn=collate_fn,
    shuffle=False
)

Model = getattr(__import__('models.' + args.model, fromlist=['Model']), 'Model')
model = Model(args)
loss_func = torch.nn.CrossEntropyLoss(weight=VALENCE_WEIGHTS)
optimizer = optim.RAdam(model.parameters(), lr=args.learning_rate)

model = model.to(args.device)
loss_func = loss_func.to(args.device)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc',
        'F1'
    ]:
        metrics[f'{stage}_{metric}'] = 0

metric_before = {}
metric_mean = {}
early_stopping_patience = 0

state = {
    'train_loss': -1.0,
    'test_loss': -1.0,
    'best_loss': -1.0,
    'train_acc': -1.0,
    'test_acc': -1.0,
    'best_acc': -1.0,
    'train_F1': -1.0,
    'test_F1': -1.0,
    'validation_loss': -1.0,
    'validation_acc': -1.0,
    'validation_F1': -1.0,
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
                validate(model, dataloader_validation, args, loss_func, state, CsvUtils2)
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
            validate(model, dataloader_validation, args, loss_func, state, CsvUtils2)
            break

    for data_loader in [dataloader_train, dataloader_test]:

        metrics_batch = {key: [] for key in metrics.keys()}

        stage = 'train'
        model.train()

        if data_loader == dataloader_test:
            stage = 'test'
            model.eval()

        conf_matrix = numpy.zeros((args.class_count, args.class_count))
        for x, y, lengths in data_loader:

            y = y.to(args.device)
            x = x.to(args.device)

            y_prim = model.forward(x, lengths)

            loss = loss_func.forward(y_prim, y)

            if data_loader == dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            y_prim_idxes = torch.argmax(y_prim, dim=1)
            acc = torch.mean((y == y_prim_idxes) * 1.0)

            F1_score = f1_score(y.to('cpu'), y_prim_idxes.to('cpu'), average='micro', zero_division=0)

            metrics_batch[f'{stage}_loss'].append(loss.cpu().item())  # Tensor(0.1) => 0.1f
            metrics_batch[f'{stage}_acc'].append(acc.cpu().item())
            metrics_batch[f'{stage}_F1'].append(F1_score)

            if stage == 'test':
                for idx, y_prim_idx in enumerate(y_prim_idxes):
                    y_idx = y[idx]
                    conf_matrix[y_idx.item(), y_prim_idx.item()] += 1

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
    state['train_F1'] = metrics_epoch['train_F1'][-1]
    state['test_F1'] = metrics_epoch['test_F1'][-1]

    # state['validation_loss'] = metrics_epoch['validation_loss'][-1]
    # state['validation_acc'] = metrics_epoch['validation_acc'][-1]
    # state['validation_F1'] = metrics_epoch['validation_F1'][-1]

    if epoch == 0:
        state['best_loss'] = metrics_epoch['test_loss'][-1]
        state['best_acc'] = metrics_epoch['test_acc'][-1]
    if state['test_loss'] < state['best_loss']:
        state['best_loss'] = metrics_epoch['test_loss'][-1]
    if state['test_acc'] > state['best_acc']:
        state['best_acc'] = metrics_epoch['test_acc'][-1]
    state['percent_improvement'] = metrics_epoch['percent_improvement'][-1]

    CsvUtils2.add_hparams(
        path_sequence=path_sequence,
        run_name=args.run_name,
        args_dict=args.__dict__,
        metrics_dict=state,
        global_step=epoch
    )

    plt.clf()
    plt.figure(figsize=(6, 7))
    plt.tight_layout()
    plt.title(f'conf_matrix epoch: {epoch}')
    plt.imshow(conf_matrix.transpose(), interpolation='nearest', cmap=plt.get_cmap('Greys'))
    titles_x = np.arange(args.class_count).astype(int)
    titles_y = np.arange(args.class_count).astype(int)
    for i in range(len(titles_y)):
        for j in range(len(titles_y)):
            plt.annotate(
                str(round(conf_matrix[i, j], 1)),
                xy=(i, j),
                horizontalalignment='center',
                verticalalignment='center',
                backgroundcolor='white'
            )
    plt.xticks(titles_x, titles_y, rotation=45)
    plt.yticks(titles_x, titles_y)
    plt.xlabel('Faktiskā klase')
    plt.ylabel('Piešķirtā klase')
    plt.savefig(f'{path_run}/{epoch}-conf.png')
    plt.close()

validate(model, dataloader_validation, args, loss_func, state, CsvUtils2)

    # plt1 = plt.plot(metrics_epoch['train_loss'], '-b', label='train loss')
    # plt2 = plt.plot(metrics_epoch['train_acc'], '--r', label='train acc')
    # plt3 = plt.plot(metrics_epoch['test_loss'], '-.c', label='test loss')
    # plt4 = plt.plot(metrics_epoch['train_acc'], '-m', label='test acc')
    #
    # plts = plt1 + plt2 + plt3 + plt4
    # plt.legend(plts, [it.get_label() for it in plts])
    # plt.draw()
    # plt.pause(0.1)

