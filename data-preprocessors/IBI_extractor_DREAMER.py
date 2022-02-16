from scipy.io import loadmat
import numpy as np
import json
import biosppy.signals.ecg as ecg
import biosppy.signals.tools as tools
from tqdm import tqdm
import argparse
from modules.dataset_utils import DatasetUtils

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-DATA_PATH', default=f'../datasets/DREAMER/data/', type=str)
parser.add_argument('-OUTPUT_PATH', default=f'../data_processed_full/', type=str)
parser.add_argument('-FREQ', default=256, type=int)  # Sampling freq of the dataset, don't change this
parser.add_argument('-CLASS_COUNT', default=2, type=int)  # [2]
parser.add_argument('-SLIDING_WINDOW', default=30, type=int)
parser.add_argument('-TIMESTEP', default=1, type=int)
parser.add_argument('-MAX_SEQ_LEN', default=55, type=int)  # 55 for 30 sec / 38 for 20 sec / 19 for 10 sec
parser.add_argument('-NORMALIZATION_TYPE', default='standard_score', type=str)  # [minmax; standard_score]
parser.add_argument('-NORMALIZATION_SCOPE', default='byperson', type=str)  # [byperson; byseq]
args, other_args = parser.parse_known_args()

person_id = []
valence_list = []
arousal_list = []
lengths = []

mmap_path = args.OUTPUT_PATH \
            + f'DREAMER_IBI_{args.SLIDING_WINDOW}sec_{args.CLASS_COUNT}C_' \
              f'{args.NORMALIZATION_TYPE}_{args.NORMALIZATION_SCOPE}.mmap'
memmap_file = np.memmap(mmap_path, dtype='float16', mode='w+', shape=(1,))
memmap_file.flush()
memmap_idx = 0

person = 1
failed = 0

# Examples how to access data:
# DATA['DREAMER']['Data'][0,0][0,11]['ScoreValence'][0][0][17][0] 11. dalībnieka 17. video valence
# DATA['DREAMER']['Data'][0,0][0,11]['ECG'][0,0]['stimuli'][0,0][17,0][:, 0] 11. dalībnieka 17. video 1. ECG lead

dataset = loadmat(args.DATA_PATH + 'DREAMER.mat')
data = dataset['DREAMER']['Data'][0, 0]
del dataset

for participant in range(0, 23):  # 23 participants
    print(f'Starting to process {participant}. participant \n')
    for video in tqdm(range(0, 18)):  # 18 video clips

        arousal = data[0, participant]['ScoreArousal'][0][0][video][0]
        valence = data[0, participant]['ScoreValence'][0][0][video][0]

        if arousal > 2.5:
            arousal = 1
        else:
            arousal = 0
        if valence > 2.5:
            valence = 1
        else:
            valence = 0

        LeadII = data[0, participant]['ECG'][0, 0]['stimuli'][0, 0][video, 0][:, 0]
        # LeadI is also available

        if np.isnan(np.max(LeadII)):
            failed = failed + 1
            print(f'LeadII data contains nan in {participant}. participants, {video}. video')
            continue

        try:
            order = int(0.3 * args.FREQ)
            filtered, _, _ = tools.filter_signal(signal=LeadII,
                                              ftype='FIR',
                                              band='bandpass',
                                              order=order,
                                              frequency=[3, 45],
                                              sampling_rate=args.FREQ)
            rpeaks, = ecg.hamilton_segmenter(signal=filtered, sampling_rate=args.FREQ)

            # correct R-peak locations
            rpeaks, = ecg.correct_rpeaks(signal=filtered,
                                 rpeaks=rpeaks,
                                 sampling_rate=args.FREQ,
                                 tol=0.05)
            length = len(LeadII)
            T = (length - 1) / args.FREQ
            ts = np.linspace(0, T, length, endpoint=False)
        except:
            print(f'Failed to filter signal for {participant}. participants {video}. video')
            failed = failed + 1

        window_count = int((int(np.max(ts)) - args.SLIDING_WINDOW + args.TIMESTEP) / args.TIMESTEP)

        window_start_time = 0
        ibi = []
        for window in range(0, window_count - 1):
            for idx in range(0, len(ts[rpeaks]) - 1):
                if window_start_time > ts[rpeaks][idx]:
                    continue
                elif window_start_time <= ts[rpeaks][idx] < window_start_time + args.SLIDING_WINDOW:
                    ibi.append(ts[rpeaks][idx + 1] - ts[rpeaks][idx])
                else:
                    break

            memmap_file = np.memmap(mmap_path, dtype='float16',
                                    mode='r+', shape=(args.MAX_SEQ_LEN,), offset=2 * memmap_idx)

            j = 0
            for val in ibi:
                memmap_file[j] = val
                j = j + 1
                if j - 1 > args.MAX_SEQ_LEN:
                    print(f'Max sequence lenght exceeded in person {person}, video {video}, window {window}')
                    break

            if len(ibi) < args.MAX_SEQ_LEN:  # Creating padding
                for i in range(0, args.MAX_SEQ_LEN - len(ibi)):
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
if args.NORMALIZATION_TYPE == 'minmax':
    if args.NORMALIZATION_SCOPE == 'byperson':
        DatasetUtils.normalization_minmax_by_person(
            memmap_path=mmap_path,
            shape=tuple([len(lengths), args.MAX_SEQ_LEN]),
            person_ids=person_id
        )
    elif args.NORMALIZATION_SCOPE == 'byseq':
        DatasetUtils.normalization_minmax_by_sample(
            memmap_path=mmap_path,
            shape=tuple([len(lengths), args.MAX_SEQ_LEN])
        )
elif args.NORMALIZATION_TYPE == 'standard_score':
    if args.NORMALIZATION_SCOPE == 'byperson':
        DatasetUtils.normalization_standard_score_by_person(
            memmap_path=mmap_path,
            shape=tuple([len(lengths), args.MAX_SEQ_LEN]),
            person_ids=person_id
        )
    elif args.NORMALIZATION_SCOPE == 'byseq':
        DatasetUtils.normalization_standard_score_by_sample(
            memmap_path=mmap_path,
            shape=tuple([len(lengths), args.MAX_SEQ_LEN])
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

json_path = mmap_path.replace('.mmap', '.json')
with open(json_path, 'w+') as json_file:
    json.dump(json_dict, json_file, indent=4)

print(f'{failed} videos have nan in ecg readings')
print(f'{np.max(lengths)} is the longest seq')

