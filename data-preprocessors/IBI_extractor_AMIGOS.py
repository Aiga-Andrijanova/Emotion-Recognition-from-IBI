from scipy.io import loadmat
import numpy as np
import os
import json
import biosppy.signals.ecg as ecg
from tqdm import tqdm
import argparse
from modules.dataset_utils import DatasetUtils

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-DATA_PATH', default=f'D:/Universitātes darbi/Bakalaura darbs/Datasets/AMIGOS/Data_Preprocessed/', type=str)
parser.add_argument('-JSON_PATH', default=f'../data/AMIGOS_IBI_30sec_byperson_small.json', type=str)
parser.add_argument('-MEMMAP_PATH', default=f'../data/AMIGOS_IBI_30sec_byperson_small.mmap', type=str)
parser.add_argument('-FREQ', default=128, type=int)
parser.add_argument('-SLIDING_WINDOW', default=30, type=int)
parser.add_argument('-TIMESTEP', default=1, type=int)
parser.add_argument('-MAX_SEQ_LEN', default=61, type=int) #61
args, other_args = parser.parse_known_args()

person_id = []
valence_list = []
arousal_list = []
lengths = []

memmap_file = np.memmap(args.MEMMAP_PATH, dtype='float16', mode='w+', shape=(1,))
memmap_file.flush()
memmap_idx = 0

person = 1
failed = 0
for filename in os.listdir(args.DATA_PATH):

    DATA = loadmat(args.DATA_PATH + filename)

    # There are 20 videos, 0-15 short videos, 16-19 long
    for video in tqdm(range(0, 15)):

        # participants 8, 24 and 28 did not attend long sessions (from the paper)
        # participant 32 doesn't have self assessment data for long videos (unknown reason)
        # seems that they forgot to write this in paper because SelfAssement.xcls
        # is just missing 32nd participant
        if video >= 16:
            if person in [8, 24, 28, 32]:
                continue

        # 9 classes
        # arousal = int(round(DATA['labels_selfassessment'][0][video][0, 0], 0)) - 1
        # valence = int(round(DATA['labels_selfassessment'][0][video][0, 1], 0)) - 1

        # 2 classes
        arousal = round(DATA['labels_selfassessment'][0][video][0, 0], 1) - 1
        valence = round(DATA['labels_selfassessment'][0][video][0, 1], 1) - 1
        if arousal > 3.5:
            arousal = 1
        else:
            arousal = 0
        if valence > 3.5:
            valence = 1
        else:
            valence = 0

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

            if len(ibi) < 20:
                ibi.clear()
                window_start_time = window_start_time + args.TIMESTEP
                continue

            memmap_file = np.memmap(args.MEMMAP_PATH, dtype='float16',
                                    mode='r+', shape=(args.MAX_SEQ_LEN,), offset=2 * memmap_idx)

            j = 0
            for val in ibi:
                memmap_file[j] = val
                j = j + 1
                if j-1 > args.MAX_SEQ_LEN:
                    print(f'Max sequence lenght exceeded in person {person}, video {video}, window {window}')
                    break

            if len(ibi) < args.MAX_SEQ_LEN:  # Creating padding
                for i in range(0, args.MAX_SEQ_LEN-len(ibi)):
                    memmap_file[j] = 0
                    j = j + 1

            memmap_file.flush()

            memmap_idx = memmap_idx + args.MAX_SEQ_LEN

            person_id.append(int(person))
            arousal_list.append(int(arousal))
            valence_list.append(int(valence))
            lengths.append(int(len(ibi)))

            ibi.clear()

            window_start_time = window_start_time + args.TIMESTEP

    person = person + 1

# WEIGHT CALCULATION
arousal_weights, _ = DatasetUtils.weight_calculation(feature_list=arousal_list)
valence_weights, class_count = DatasetUtils.weight_calculation(feature_list=valence_list)
# WEIGHT CALCULATION

# DATA NORMALIZATION
DatasetUtils.normalization_minmax_by_person(
    memmap_path=args.MEMMAP_PATH,
    shape=tuple([len(lengths), args.MAX_SEQ_LEN]),
    people=person_id
)
# DATA NORMALIZATION

json_dict = dict()
json_dict["shape"] = [len(lengths), args.MAX_SEQ_LEN]
json_dict["person_id"] = person_id
json_dict["class_count"] = class_count
json_dict["arousal"] = arousal_list
json_dict["valence"] = valence_list
json_dict["lengths"] = lengths
json_dict["valence_weights"] = valence_weights
json_dict["arousal_weights"] = arousal_weights

with open(args.JSON_PATH, 'w+') as json_file:
    json.dump(json_dict, json_file, indent=4)

print(f'{failed} videos have nan in ecg readings')
print(f'{np.max(lengths)} is the longest seq')
