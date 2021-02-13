from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
import os
import json
import biosppy.signals.ecg as ecg
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-DATA_PATH', default=f'D:/Universitātes darbi/Bakalaura darbs/Datasets/AMIGOS/Data_Preprocessed/', type=str)
parser.add_argument('-JSON_PATH', default=f'Data/AllIBIdata.json', type=str)
parser.add_argument('-MEMMAP_PATH', default=f'Data/AllIBIdata.memmap', type=str)
parser.add_argument('-FREQ', default=128, type=int)
parser.add_argument('-SLIDING_WINDOW', default=10, type=int)
parser.add_argument('-TIMESTEP', default=1, type=int)
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

        arousal = DATA['labels_selfassessment'][0][video][0, 0]
        valence = DATA['labels_selfassessment'][0][video][0, 1]

        LeadII = DATA['joined_data'][0][video][:, 14]
        LeadIII = DATA['joined_data'][0][video][:, 15]

        if np.isnan(np.max(LeadIII)):
            failed = failed + 1
            continue

        ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate \
            = ecg.ecg(signal=LeadIII, sampling_rate=args.FREQ, show=False)

        window_count = int((int(np.max(ts)) - args.SLIDING_WINDOW + args.TIMESTEP) / args.TIMESTEP)

        window_start_time = 0
        ibi = []
        for window in range(0, window_count-1):
            for idx in range(0, len(ts[rpeaks])-1):
                if window_start_time > ts[rpeaks][idx]:
                    continue
                elif window_start_time <= ts[rpeaks][idx] < window_start_time + args.SLIDING_WINDOW:
                    ibi.append(ts[rpeaks][idx+1] - ts[rpeaks][idx])
                else:
                    break

            memmap_file = np.memmap(args.MEMMAP_PATH, dtype='float16',
                                    mode='r+', shape=(len(ibi),), offset=2 * memmap_idx)

            j = 0
            for val in ibi:
                memmap_file[j] = val
                j = j + 1

            memmap_file.flush()

            memmap_idx = memmap_idx + len(ibi)

            person_id.append(int(person))
            arousal_list.append(float(arousal))
            valence_list.append(float(valence))
            lenghts.append(int(len(ibi)))

            ibi.clear()

            window_start_time = window_start_time + args.TIMESTEP

    person = person + 1

json_dict = dict()
json_dict["person_id"] = person_id
json_dict["arousal"] = arousal_list
json_dict["valence"] = valence_list
json_dict["lenghts"] = lenghts

with open(args.JSON_PATH, 'w+') as json_file:
    json.dump(json_dict, json_file, indent=4)

print(f'{failed} videos have nan in ecg readings')
