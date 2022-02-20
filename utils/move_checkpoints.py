import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle 
from pathlib import Path
import os
from shutil import copy2
from pathlib import Path
import re

directories = ['logdir']
snapshots_directory = 'snapshots'
try:
    snapshots_dir = os.path.join(os.getcwd(), snapshots_directory)
    os.mkdir(snapshots_dir)
except:
    pass

id_ranges = []

ids = [(355, 357)]
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
        if len(exp_id) > 4 and exp_id[0] == 'S' and exp_id[4] == '_':
            if int(exp_id[1:4]) in id_ranges:
                print(exp_id, flush=True)
                models = os.path.join(experiment, 'models')
                latest_ep = 0
                for ep in os.listdir(models):
                    if ep is None or 'ep_' not in ep:
                        continue 
                    cur_episode_step = int(re.search(r'\d+', ep).group())
                    if cur_episode_step > latest_ep:
                        latest_ep = cur_episode_step
                    
                
                if latest_ep == 0:
                    continue
                latest_ep = str(latest_ep)
                alice_checkpoints = os.listdir(models +'/ep_'+latest_ep + '/alice')
                bob_checkpoints = os.listdir(models +'/ep_'+latest_ep + '/bob')

                for file in alice_checkpoints:
                    results_folder = 'alice_{}_ep{}'.format(exp_id, latest_ep)
                    try:
                        os.mkdir(os.path.join(snapshots_dir, results_folder))
                    except:
                        pass

                    copy2(os.path.join(models +'/ep_'+latest_ep + '/alice', file), snapshots_dir + '/' + results_folder)

                for file in bob_checkpoints:
                    results_folder = 'bob_{}_ep{}'.format(exp_id, latest_ep)
                    try:
                        os.mkdir(os.path.join(snapshots_dir, results_folder))
                    except:
                        pass

                    copy2(os.path.join(models +'/ep_'+latest_ep + '/bob', file), snapshots_dir + '/' + results_folder)
