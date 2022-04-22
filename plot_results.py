import os
import glob

import json
import numpy as np
import pandas as pd

relevant_files = "./*/*/*/training_dynamics.csv"
files = glob.glob(relevant_files)
print(len(files), files[:3])

output_dict = {}
for f in files:
    print("Reading file:", f)
    df = pd.read_csv(f)
    final_results = df.tail(1)
    train_mse, test_mse = float(final_results["train_loss"]), float(final_results["test_loss"])
    print(final_results, train_mse, test_mse)
    
    # Get config from the model file
    file_parts = f.split(os.sep)
    config = file_parts[-3].split("_")  # file, model, config, outputs
    print(config)
    
    assert config[-1] == "pos"
    assert config[-3] == "k"
    k = int(config[-2])
    
    assert config[-6] == "train"
    assert config[-5] == "ex"
    num_examples = int(config[-4])
    print(f"k: {k} / num examples: {num_examples}")
    
    if num_examples not in output_dict:
        output_dict[num_examples] = {}
    assert k not in output_dict[num_examples]
    output_dict[num_examples][k] = (train_mse, test_mse)

print(json.dumps(output_dict, indent=4, sort_keys=True))

# TODO: Plot the dictionary here

# TODO: Plot the intervention results here
