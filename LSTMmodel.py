import json
import numpy as np
import torch
import torch.utils.data


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
                data_json['lenghts'][idx_sample],  # I misspelled lengths in the json file ¯\_(ツ)_/¯
                data_json['arousal'][idx_sample],
                data_json['valence'][idx_sample]]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ibi, lengths, arousal, valance = self.data[idx]
        t_ibi = torch.FloatTensor(ibi)
        t_lengths = torch.FloatTensor(lengths)
        t_arousal = torch.FloatTensor(arousal)
        t_valance = torch.FloatTensor(valance)
        return t_ibi, t_lengths, t_arousal, t_valance


torch.manual_seed(0)  # lock seed
dataset_full = DatasetIBI('Data/AllIBIdata.json')
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full, lengths=[int(len(dataset_full)*0.8), len(dataset_full)-int(len(dataset_full)*0.8)])

torch.seed()  # init random seed


def collate_fn(batch):
    t_ibi, t_lengths, t_arousal, t_valance = zip(*batch)
    # sorted, indices = torch.sort(t_lengths, descending=True)
    # t_ibi[indices]
    indices = torch.argsort(t_lengths, dim=0, descending=True)
    t_ibi = t_ibi[indices]
    # t_ibi # (B, Max_seq, F) padded by zeros at end, MUST be sorted by longest seq first
    #  (B, )
    t_ibi = t_ibi[None, :, :]
    t_lengths = t_lengths[None, :]
    t_arousal = t_arousal[None, :]
    t_valance = t_valance[None, :]
    return t_ibi, t_lengths, t_arousal, t_valance


dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    collate_fn=collate_fn,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    collate_fn=collate_fn,
    shuffle=False
)

# Padodot uz LSTM bus vajadzigs PackedSequence
#  (https://www.notion.so/evalds/PackedSequence-RNN-5922eaeea48644a2b96fe5e13ef1a185)