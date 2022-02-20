import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle 
from pathlib import Path
import os
from shutil import copy2

directories = ['logdir']
results_folder = 'eval_pkls_2'

try:
    os.mkdir(os.path.join(os.getcwd(), results_folder))
except:
    pass

for directory in directories:
    directory = Path(directory)
    if not directory.exists():
        continue
    # Loop through files
    for experiment in directory.iterdir():
        try:
            exp_id = str(experiment).split('/')[-1]
        except:
            pass
        if len(exp_id) > 4 and exp_id[0] == 'S' and exp_id[4] == '_':
            if int(exp_id[1:4]) >= 337:
                for file in os.listdir(experiment):
                    if file.endswith(".pkl"):
                        copy2(os.path.join(experiment, file), os.getcwd() + '/' + results_folder)
