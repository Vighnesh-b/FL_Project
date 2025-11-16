import os
import time
import torch
from glob import glob
from utils.fed_utils import fed_avg
from models.unetr_model import get_unetr  # adjust path if needed
import config
from config import TRAIN_SIZES


class FederatedServer:
    def __init__(self, model_fn, shared_dir="shared", num_clients=4, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_fn(self.device)
        self.shared_dir = shared_dir
        self.global_dir = os.path.join(shared_dir, "global")
        self.updates_dir = os.path.join(shared_dir, "updates")
        self.num_clients = num_clients

        os.makedirs(self.global_dir, exist_ok=True)
        os.makedirs(self.updates_dir, exist_ok=True)

    def save_global_model(self, round_num):
        path = os.path.join(self.global_dir, f"global_round_{round_num}.pth")
        torch.save(self.model.state_dict(), path)
        latest = os.path.join(self.global_dir, "global_latest.pth")
        torch.save(self.model.state_dict(), latest)
        print(f"[Server] Saved global model (Round {round_num})")

    def wait_for_all_clients(self, round_num):
        print(f"[Server] Waiting for {self.num_clients} clients to upload updates for Round {round_num}...")
        while True:
            update_files = glob(os.path.join(self.updates_dir, "client_*_weights.pth"))
            if len(update_files) >= self.num_clients:
                print(f"[Server] All {self.num_clients} updates received.")
                return update_files
            time.sleep(5)

    def aggregate(self, client_files):
        print(f"[Server] Aggregating {len(client_files)} client models...")
        state_dicts = [torch.load(file, map_location=self.device) for file in client_files]
        new_state = fed_avg(state_dicts,TRAIN_SIZES)
        self.model.load_state_dict(new_state)
        print("[Server] Aggregation complete.")
        return new_state

    def clear_client_updates(self):
        for file in glob(os.path.join(self.updates_dir, "client_*_weights.pth")):
            os.remove(file)
        print("[Server] Cleared client updates.\n")

    def initialize_global_model(self):
        """Initialize and save the initial global model before first round."""
        latest = os.path.join(self.global_dir, "global_latest.pth")
        if not os.path.exists(latest):
            torch.save(self.model.state_dict(), latest)
            print("[Server] Initialized and saved initial global model.")
        else:
            print("[Server] Initial global model already exists.")

    def run_round(self, round_num):
        client_files = self.wait_for_all_clients(round_num)
        self.aggregate(client_files)
        self.save_global_model(round_num)
        self.clear_client_updates()
        print(f"[Server] Round {round_num} complete.\n")


if __name__ == "__main__":
    CURRENT_ROUND = getattr(config, "START_ROUND", 1)
    NUM_ROUNDS = getattr(config, "NUM_ROUNDS", 3)
    NUM_CLIENTS = getattr(config, "NUM_CLIENTS", 4)

    server = FederatedServer(model_fn=get_unetr, shared_dir="shared", num_clients=NUM_CLIENTS)
    
    # Initialize global model before starting rounds
    server.initialize_global_model()

    for r in range(CURRENT_ROUND, NUM_ROUNDS + 1):
        print(f"\n====================== ROUND {r} ======================")
        server.run_round(r)

    print(f"\nTraining complete after {NUM_ROUNDS} rounds.\n")
