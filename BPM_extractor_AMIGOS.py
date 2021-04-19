from scipy.io import loadmat
import numpy as np
import os
import json
import biosppy.signals.ecg as ecg
from tqdm import tqdm
import argparse
from collections import Counter

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-DATA_PATH', default=f'D:/Universitātes darbi/Bakalaura darbs/Datasets/AMIGOS/Data_Preprocessed/', type=str)
parser.add_argument('-JSON_PATH', default=f'data/AllBPMdata.json', type=str)
parser.add_argument('-MEMMAP_PATH', default=f'data/AllBPMdata.mmap', type=str)
parser.add_argument('-FREQ', default=128, type=int)
parser.add_argument('-SLIDING_WINDOW', default=10, type=int)
parser.add_argument('-TIMESTEP', default=1, type=int)
parser.add_argument('-MAX_SEQ_LEN', default=21, type=int)
args, other_args = parser.parse_known_args()

person_id = []
valence_list = []
arousal_list = []
lenghts = []

memmap_file = np.memmap(args.MEMMAP_PATH, dtype='float16', mode='w+', shape=(1,))
memmap_file.flush()
memmap_idx = 0

person = 1
failed = 0
for filename in os.listdir(args.DATA_PATH):

    DATA = loadmat(args.DATA_PATH + filename)

    # There are 20 videos, 0-15 short videos, 16-19 long
    for video in tqdm(range(0, 15)):

        # converting to 9 classes and [1;9] -> [0;8]
        arousal = int(round(DATA['labels_selfassessment'][0][video][0, 0], 0)) - 1
        valence = int(round(DATA['labels_selfassessment'][0][video][0, 1], 0)) - 1

        LeadII = DATA['joined_data'][0][video][:, 14]
        LeadIII = DATA['joined_data'][0][video][:, 15]

        if np.isnan(np.max(LeadIII)):
            failed = failed + 1
            continue

        ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = \
            ecg.ecg(signal=LeadIII, sampling_rate=args.FREQ, show=False)

        window_count = int((int(np.max(ts)) - args.SLIDING_WINDOW + args.TIMESTEP) / args.TIMESTEP)

        window_start_time = 0
        hr = []
        for window in range(0, window_count-1):
            for idx in range(0, len(heart_rate_ts)):
                if window_start_time > heart_rate_ts[idx]:
                    continue
                elif window_start_time <= heart_rate_ts[idx] < window_start_time + args.SLIDING_WINDOW:
                    hr.append(heart_rate[idx])
                else:
                    break

            memmap_file = np.memmap(args.MEMMAP_PATH, dtype='float16',
                                    mode='r+', shape=(args.MAX_SEQ_LEN,), offset=2 * memmap_idx)

            j = 0
            for val in hr:
                memmap_file[j] = val
                j = j + 1
                if j-1 > args.MAX_SEQ_LEN:
                    print(f'Max sequence lenght exceeded in person {person}, video {video}, window {window}')
                    break

            if len(hr) < args.MAX_SEQ_LEN:  # Creating padding
                for i in range(0, args.MAX_SEQ_LEN-len(hr)):
                    memmap_file[j] = 0
                    j = j + 1

            memmap_file.flush()

            memmap_idx = memmap_idx + args.MAX_SEQ_LEN

            person_id.append(int(person))
            arousal_list.append(float(arousal))
            valence_list.append(float(valence))
            lenghts.append(int(len(hr)))

            hr.clear()

            window_start_time = window_start_time + args.TIMESTEP

    person = person + 1

# WEIGHT CALCULATION
total = len(arousal_list)

Counter_arousal_dict = Counter(arousal_list)
arousal_weights = []
sorted_arousal_dict = {key: val for key, val in sorted(Counter_arousal_dict.items(), key=lambda ele: ele[0])}
for key in sorted_arousal_dict.keys():
    arousal_weights.append((1 / sorted_arousal_dict[key]) * total / 2.0)

Counter_valence_dict = Counter(valence_list)
valence_weights = []
sorted_valence_dict = {key: val for key, val in sorted(Counter_valence_dict.items(), key=lambda ele: ele[0])}
for key in sorted_valence_dict.keys():
    valence_weights.append((1 / sorted_valence_dict[key]) * total / 2.0)
# WEIGHT CALCULATION

# DATA NORMALIZATION
memmap_file = np.memmap(args.MEMMAP_PATH, dtype='float16',
                                    mode='r+', shape=len(lenghts) * args.MAX_SEQ_LEN)
max = np.max(memmap_file)
min = np.min(memmap_file[np.nonzero(memmap_file)])
delta = max - min
for i in range(len(lenghts) * args.MAX_SEQ_LEN-1):
    if memmap_file[i] != 0:
        memmap_file[i] = (memmap_file[i] - min) / delta
memmap_file.flush()
# DATA NORMALIZATION

json_dict = dict()
json_dict["shape"] = [len(lenghts), args.MAX_SEQ_LEN]
json_dict["class_count"] = len(sorted_valence_dict.keys())
json_dict["person_id"] = person_id
json_dict["arousal"] = arousal_list
json_dict["valence"] = valence_list
json_dict["lengths"] = lenghts
json_dict["valence_weights"] = valence_weights
json_dict["arousal_weights"] = arousal_weights

with open(args.JSON_PATH, 'w+') as json_file:
    json.dump(json_dict, json_file, indent=4)

print(f'{failed} videos have nan in ecg readings')
print(f'{np.max(lenghts)} is the longest seq')
