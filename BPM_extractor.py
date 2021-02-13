from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
import os
import json
import biosppy.signals.ecg as ecg
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-PATH', default=f'D:/Universitātes darbi/Bakalaura darbs/Datasets/AMIGOS/Data_Test/', type=str)
parser.add_argument('-FREQ', default=128, type=int)
parser.add_argument('-SLIDING_WINDOW', default=10, type=int)
parser.add_argument('-TIMESTEP', default=1, type=int)
args, other_args = parser.parse_known_args()

person_id = []
valence_list = []
arousal_list = []
lenghts = []

memmap_file = np.memmap('data.memmap', dtype='float16', mode='w+', shape=(1,))
memmap_file.flush()
memmap_idx = 0

person = 1
failed = 0
for filename in os.listdir(args.PATH):

    DATA = loadmat(args.PATH + filename)

    # There are 20 videos, 0-15 short videos, 16-19 long
    for video in tqdm(range(0, 1)):

        arousal = DATA['labels_selfassessment'][0][video][0, 0]
        valence = DATA['labels_selfassessment'][0][video][0, 1]

        LeadII = DATA['joined_data'][0][video][:, 14]
        LeadIII = DATA['joined_data'][0][video][:, 15]

        if np.isnan(np.max(LeadIII)):
            failed = failed + 1
            break

        ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(signal=LeadIII, sampling_rate=args.FREQ, show=False)

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

            memmap_file = np.memmap('data.memmap', dtype='float16', mode='r+', shape=(len(hr),), offset=2 * memmap_idx)

            j = 0
            for val in hr:
                memmap_file[j] = val
                j = j + 1

            memmap_file.flush()

            memmap_idx = memmap_idx + len(hr)

            person_id.append(int(person))
            arousal_list.append(float(arousal))
            valence_list.append(float(valence))
            lenghts.append(int(len(hr)))

            hr.clear()

            window_start_time = window_start_time + args.TIMESTEP

    person = person + 1

json_dict = dict()
json_dict["person_id"] = person_id
json_dict["arousal"] = arousal_list
json_dict["valence"] = valence_list
json_dict["lenghts"] = lenghts

with open('data.json', 'w+') as json_file:
    json.dump(json_dict, json_file, indent=4)

print(f'{failed} videos have nan in ecg readings')
