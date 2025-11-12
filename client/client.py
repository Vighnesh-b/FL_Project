import os
import time
import csv
import torch
from datetime import datetime
from models.unetr_model import get_unetr
from utils.train_utils import train_one_epoch, evaluate, combined_loss
import config


class FederatedClient:
    def __init__(self, client_id, model_fn, data, shared_dir="shared", device=None):
        self.client_id = client_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_fn(self.device)
        self.shared_dir = shared_dir
        self.global_path = os.path.join(shared_dir, "global", "global_latest.pth")
        self.update_path = os.path.join(shared_dir, "updates", f"client_{client_id}_weights.pth")

        # Logging setup
        self.log_dir = os.path.join(shared_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"client_{client_id}_log.csv")
        self._init_log_file()

        # data already includes dataloaders
        self.train_loader, self.val_loader, self.test_loader = data

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = combined_loss

    def _init_log_file(self):
        """Initialize the CSV log file with header if not present."""
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "round", "epoch", "train_loss", "val_loss", "val_dice"])

    def _log_metrics(self, round_num, epoch, train_loss, val_loss, val_dice):
        """Append training/validation metrics to CSV."""
        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round_num, epoch, 
                f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_dice:.4f}"
            ])

    def load_global_model(self):
        while not os.path.exists(self.global_path):
            print(f"[Client {self.client_id}] Waiting for global model...")
            time.sleep(5)
        self.model.load_state_dict(torch.load(self.global_path, map_location=self.device))
        print(f"[Client {self.client_id}] Loaded global model from server.")

    def train_local(self, round_num, epochs=config.EPOCHS_PER_CLIENT):
        print(f"[Client {self.client_id}] Starting local training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(self.model, self.train_loader, self.optimizer, self.criterion, self.device)
            val_loss, val_dice = evaluate(self.model, self.val_loader, self.criterion, self.device)

            print(f"[Client {self.client_id}] Epoch {epoch}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}")

            self._log_metrics(round_num, epoch, train_loss, val_loss, val_dice)

        print(f"[Client {self.client_id}] Local training complete for Round {round_num}.")

    def upload_model(self):
        torch.save(self.model.state_dict(), self.update_path)
        print(f"[Client {self.client_id}] Uploaded weights to {self.update_path}")

    def run(self):
        for round_num in range(config.START_ROUND, config.NUM_ROUNDS + 1):
            print(f"\n[Client {self.client_id}] ===== Round {round_num} =====")
            self.load_global_model()
            self.train_local(round_num)
            self.upload_model()

            print(f"[Client {self.client_id}] Waiting for next round aggregation...\n")
            while os.path.exists(self.update_path):
                print(f"[Client {self.client_id}] Waiting for server to aggregate round {round_num}...")
                time.sleep(10)


if __name__ == "__main__":
    from datasets.brain_tumor_dataset import get_client_data 

    CLIENT_ID = int(os.getenv("CLIENT_ID", 1))
    data = get_client_data(CLIENT_ID)

    client = FederatedClient(client_id=CLIENT_ID, model_fn=get_unetr, data=data)
    client.run()
