import os
import copy
import torch
import pandas as pd
from glob import glob

# --- FedAvg (weighted by dataset size) ---
def fed_avg(state_dicts, data_sizes):

    total_size = sum(data_sizes)
    avg_state = copy.deepcopy(state_dicts[0])

    for key in avg_state.keys():
        # Weighted sum
        avg_state[key] = sum(
            state_dicts[i][key] * (data_sizes[i] / total_size)
            for i in range(len(state_dicts))
        )

    return avg_state


# --- Resume Global State (checkpoint + logs) ---
def resume_global_state(global_dir, logs_dir):

    global_metrics_path = os.path.join(logs_dir, "global_metrics.csv")
    start_round = 1
    global_weights = None
    global_metrics_list = []

    # Find latest checkpoint
    ckpts = glob(os.path.join(global_dir, "global_round_*.pth"))
    ckpts.sort()

    if len(ckpts) > 0:
        latest_ckpt = ckpts[-1]
        start_round = int(latest_ckpt.split("_round_")[-1].split(".pth")[0]) + 1
        global_weights = torch.load(latest_ckpt, map_location="cpu")

    # Load existing metrics if CSV exists
    if os.path.exists(global_metrics_path):
        global_metrics_list = pd.read_csv(global_metrics_path).to_dict("records")

    return start_round, global_weights, global_metrics_list
