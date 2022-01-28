from collections import Counter
import numpy as np

class DatasetUtils():

    @staticmethod
    def weight_calculation(feature_list):
        # Calculates class weights for AMIGOS dataset

        total = len(feature_list)
        counter_feature_dict = Counter(feature_list)
        feature_weights = []
        sorted_feature_dict = {key: val for key, val in sorted(counter_feature_dict.items(), key=lambda ele: ele[0])}
        for key in sorted_feature_dict.keys():
            feature_weights.append((1 / sorted_feature_dict[key]) * total / 2.0)

        class_count = len(sorted_feature_dict.keys())
        return feature_weights, class_count

    @staticmethod
    def normalization_minmax_by_sample(memmap_path, shape, range_min=0, range_max=1.0):
        # Normalizes data by each sample for AMIGOS

        memmap_file = np.memmap(memmap_path, dtype='float16',
                                mode='r+', shape=shape)
        max_seq_len = memmap_file.shape[1]
        eps = 1e-8

        for i in range(len(memmap_file)-1):
            data_max = np.max(memmap_file[i])
            data_min = np.min(memmap_file[i])

            delta = data_max - data_min
            memmap_file[i, :max_seq_len] = (memmap_file[i, :max_seq_len] - data_min) / (delta + eps) \
                                           * (range_max - range_min) + range_min
            memmap_file.flush()

    @staticmethod
    def normalization_minmax_by_person(memmap_path, shape, person_ids, range_min=0, range_max=1.0):
        # Normalizes data by each person, assumes that memmap data is ordered by person

        memmap_file = np.memmap(memmap_path, dtype='float16',
                                mode='r+', shape=shape)
        max_seq_len = memmap_file.shape[1]
        eps = 1e-8

        person_ids.reverse()
        person_count = person_ids[0]
        person_change_idx = []
        for person in range(1,person_count):
            person_change_idx.append(person_ids.index(person))
        person_change_idx.append(0)
        person_change_idx.reverse()

        for person in range(1, len(person_change_idx)-1):
            start_idx = person_change_idx[person-1]
            end_idx = person_change_idx[person]

            data_max = np.max(memmap_file[start_idx:end_idx])
            data_min = np.min(memmap_file[start_idx:end_idx])
            delta = data_max - data_min

            memmap_file[start_idx:end_idx, :max_seq_len] = (memmap_file[start_idx:end_idx, :max_seq_len] - data_min) / \
                                                           (delta + eps) * (range_max - range_min) + range_min
            memmap_file.flush()

    @staticmethod
    def normalization_standard_score_by_sample(memmap_path, shape):
        memmap_file = np.memmap(memmap_path, dtype='float16',
                                mode='r+', shape=shape)
        max_seq_len = memmap_file.shape[1]
        eps = 1e-8

        for i in range(len(memmap_file)-1):
            data_mean = np.mean(memmap_file[i])
            data_std = np.std(memmap_file[i])

            memmap_file[i, :max_seq_len] = (memmap_file[i, :max_seq_len] - data_mean) / (data_std + eps)
            memmap_file.flush()

    @staticmethod
    def normalization_standard_score_by_person(memmap_path, shape, person_ids):
        # Normalizes data by each person, assumes that memmap data is ordered by person

        memmap_file = np.memmap(memmap_path, dtype='float16',
                                mode='r+', shape=shape)
        max_seq_len = memmap_file.shape[1]
        eps = 1e-8

        person_ids.reverse()
        person_count = person_ids[0]
        person_change_idx = []
        for person in range(1,person_count):
            person_change_idx.append(person_ids.index(person))
        person_change_idx.append(0)
        person_change_idx.reverse()

        for person in range(1, len(person_change_idx)-1):
            start_idx = person_change_idx[person-1]
            end_idx = person_change_idx[person]

            data_mean = np.mean(memmap_file[start_idx:end_idx])
            data_std = np.std(memmap_file[start_idx:end_idx])

            memmap_file[start_idx:end_idx, :max_seq_len] = (memmap_file[start_idx:end_idx, :max_seq_len] - data_mean) / \
                                                           (data_std + eps)
            memmap_file.flush()