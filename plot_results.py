import os
import glob

import numpy as np
import pandas as pd

# relevant_files = "./outputs/exp_01_train_ex_[0-9]+_k_[0-9]+_pos/models/training_dynamics.csv"
relevant_files = "./*/*/*/training_dynamics.csv"
files = glob.glob(relevant_files)
print(len(files), files[:3])

for f in files:
    print("Reading file:", f)
    df = pd.read_csv(f)
    print(df.tail(1))
    exit()
