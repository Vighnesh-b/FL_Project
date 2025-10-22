import torch
import os
from utils.fed_utils import fed_avg
from client.client import run_client
from models.unetr_model import get_unetr
from config import *

def run_server():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_weights = None
    os.makedirs(GLOBAL_MODEL_DIR, exist_ok=True)

    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\n===== ROUND {rnd} =====")
        client_weights = []

        for cid in range(1, NUM_CLIENTS + 1):
            print(f"\n-- Client {cid} --")
            local_weights = run_client(cid, global_weights, EPOCHS_PER_CLIENT, BATCH_SIZE, device)
            client_weights.append(local_weights)

        global_weights = fed_avg(client_weights)
        torch.save(global_weights, f"{GLOBAL_MODEL_DIR}/global_round_{rnd}.pth")
        print(f"Global model for round {rnd} saved.")

    print("Federated training finished!")
