import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from collections import Counter

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-dataset_path', default='Data/AllIBIdata.json', type=str)
args, other_args = parser.parse_known_args()

path_data_json = args.dataset_path
with open(path_data_json) as fp:
    data_json = json.load(fp)

data_len = []
data_arousal = []
data_valence = []

for idx_sample in range(data_json['shape'][0]):
    data_len.append(data_json['lengths'][idx_sample])
    data_arousal.append(data_json['arousal'][idx_sample])
    data_valence.append(data_json['valence'][idx_sample])

barWidth = 0.9

Counter_len_dict = Counter(data_len)
plt.subplot(1, 3, 1)
plt.bar(list(Counter_len_dict.keys()), Counter_len_dict.values(), width=barWidth, color='g')
plt.title('Data length histogram')
plt.xticks(np.sort(list(Counter_len_dict.keys())))

Counter_arousal_dict = Counter(data_arousal)
plt.subplot(1, 3, 2)
plt.bar(list(Counter_arousal_dict.keys()), Counter_arousal_dict.values(), width=barWidth, color='y')
plt.title('Data arousal histogram')
plt.xticks(np.sort(list(Counter_arousal_dict.keys())))

Counter_valence_dict = Counter(data_valence)
plt.subplot(1, 3, 3)
plt.bar(list(Counter_valence_dict.keys()), Counter_valence_dict.values(), width=barWidth, color='y')
plt.title('Data valence histogram')
plt.xticks(np.sort(list(Counter_valence_dict.keys())))

plt.show()
pass