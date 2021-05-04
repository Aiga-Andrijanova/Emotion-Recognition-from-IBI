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
    def normalization_minmax_by_sample(memmap_path, shape):
        # Normalizes data by each sample for AMIGOS

        memmap_file = np.memmap(memmap_path, dtype='float16',
                                mode='r+', shape=shape)
        max_seq_len = memmap_file.shape[1]

        for i in range(len(memmap_file)-1):
            max = np.max(memmap_file[i])
            min = np.min(memmap_file[i, np.nonzero(memmap_file[i])])

            delta = max - min
            for j in range(max_seq_len):
                if memmap_file[i, j] != 0:
                    memmap_file[i, j] = (memmap_file[i, j] - min) / delta
            memmap_file.flush()

    @staticmethod
    def normalization_minmax_by_person(memmap_path, shape, people):
        # Normalizes data by each person for AMIGOS

        memmap_file = np.memmap(memmap_path, dtype='float16',
                                mode='r+', shape=shape)
        max_seq_len = memmap_file.shape[1]

        person = 0
        for i in range(len(memmap_file) - 1):
            if person != people[i]:
                max = np.max(memmap_file[i])
                min = np.min(memmap_file[i, np.nonzero(memmap_file[i])])
                delta = max - min
                person = people[i]

            for j in range(max_seq_len):
                if memmap_file[i, j] != 0:
                    memmap_file[i, j] = (memmap_file[i, j] - min) / delta
            memmap_file.flush()
