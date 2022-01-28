import json
import numpy as np

path_data_json = 'test_data/AMIGOS_BPM_30sec_byperson.json'
with open(path_data_json) as fp:
    data_json = json.load(fp)

path_data_mmap = path_data_json.replace('.json', '.mmap')
data_mmap = np.memmap(
    path_data_mmap,
    mode='r',
    dtype=np.float16,
    shape=tuple(data_json['shape'])
)

pass