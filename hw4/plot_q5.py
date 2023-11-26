import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader


directory = "cs16831/data/"

# Get all files in the directory that start with "event_file_prefix"
dirs = [folder for folder in os.listdir(directory) if folder.startswith("hw4_q5_")]
# from every directory, get the event file
event_files = []
for folder in dirs:
    event_files += [file for file in os.listdir(os.path.join(directory, folder)) if file.startswith("events")]

print(event_files)

# Read each event file into a Pandas DataFrame
eval_returns = {}
for folder in dirs:
    event_file = [file for file in os.listdir(os.path.join(directory, folder)) if file.startswith("events")]
    file_path = os.path.join(directory, folder, event_file[0])
    event_file_reader = EventFileLoader(file_path)
    values = []
    for event in event_file_reader.Load():
        # Process each event here
        # get Eval_AverageReturn and print

        if len(event.summary.value) > 0:
            if event.summary.value[0].tag == "Eval_AverageReturn":
                values.append(event.summary.value[0].tensor.float_val[0])

    key = folder.split("hw4_q5_", 1)[1].split("_cheetah-hw4", 1)[0]
    print(key)
    # key = folder
    eval_returns[key] = values

print(eval_returns)

# make subplots to compare different hyperparameters
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(eval_returns["cheetah_cem_4"], label="cheetah_cem_4")
axs[0, 0].plot(eval_returns["cheetah_random"], label="cheetah_random")
axs[0, 0].legend()
axs[0, 0].set_xlabel("Iteration")
axs[0, 0].set_ylabel("Eval_AverageReturn")
axs[0, 0].set_title("CEM 4 vs Random")
axs[0, 1].plot(eval_returns["cheetah_cem_2"], label="cheetah_cem_2")
axs[0, 1].plot(eval_returns["cheetah_random"], label="cheetah_random")
axs[0, 1].legend()
axs[0, 1].set_xlabel("Iteration")
axs[0, 1].set_ylabel("Eval_AverageReturn")
axs[0, 1].set_title("CEM 2 vs Random")
# axs[1, 0].plot(eval_returns["reacher_horizon5"], label="reacher_horizon5")
# axs[1, 0].plot(eval_returns["reacher_horizon15"], label="reacher_horizon15")
# axs[1, 0].plot(eval_returns["reacher_horizon30"], label="reacher_horizon30")
# axs[1, 0].legend()
# axs[1, 0].set_xlabel("Iteration")
# axs[1, 0].set_ylabel("Eval_AverageReturn")
# axs[1, 0].set_title("Planning Horizon: Eval_AverageReturn vs Iteration")

plt.show()


