import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import pickle 
from pathlib import Path
import os

directories = ['logdir']
results_folder = 'eval_pkls'

try:
    os.mkdir(os.path.join(os.getcwd(), results_folder))
except:
    pass

id_ranges = []

# ids = [(82, 84)]
ids = [(16,19)]

for start, end in ids:
    id_ranges.extend(list(range(start, end+1)))
print('total ids', len(id_ranges))
for directory in directories:
    directory = Path(directory)
    if not directory.exists():
        continue
    # Loop through files
    for experiment in directory.iterdir():
        exp_id = str(experiment).split('/')[-1]
        if len(exp_id) > 4 and exp_id[0] == 'N' and exp_id[4] == '_':
            if int(exp_id[1:4]) in id_ranges:
                print(exp_id, flush=True)
                eval_scores = {} 
                tb_folder = experiment.joinpath('tb')
                try:
                    for event_file in tb_folder.iterdir():
                        for e in summary_iterator(str(event_file)):
                            for v in e.summary.value:
                                if 'eval' in v.tag:
                                    if v.tag in eval_scores.keys():
                                        eval_scores[v.tag].append([v.simple_value, e.step])
                                    else:
                                        eval_scores[v.tag] = [[v.simple_value, e.step]]
                except:
                    pass

                
                f = open(results_folder + '/' + exp_id[:5] + "evals.pkl","wb")
                pickle.dump(eval_scores,f)
                f.close()

