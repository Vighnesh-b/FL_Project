# training_api.py
# -*- coding: utf-8 -*-
"""
Final Training API Server for Federated Learning
- Fixed stop endpoint, atomic saves, aggregation retries
- CSV timestamps fixed
- Metrics display only for actual client training
- Correct client IDs in metrics
"""
print("DEBUG: training_api.py started loading")
import sys
print("DEBUG Python:", sys.version)

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import threading
from datetime import datetime
import time
import traceback
import csv

from config import (
    DEVICE as CONFIG_DEVICE,
    TRAIN_SIZES,
    NUM_CLIENTS,
    NUM_ROUNDS,
    EPOCHS_PER_CLIENT,
    LEARNING_RATE,
    START_ROUND
)
from models.unetr_model import get_unetr
from datasets.brain_tumor_dataset import get_client_data
from utils.train_utils import train_one_epoch, evaluate, combined_loss
from utils.fed_utils import fed_avg

app = Flask(__name__)
CORS(app)

SHARED_DIR = 'shared'
GLOBAL_DIR = os.path.join(SHARED_DIR, 'global')
UPDATES_DIR = os.path.join(SHARED_DIR, 'updates')
LOGS_DIR = os.path.join(SHARED_DIR, 'logs')
os.makedirs(GLOBAL_DIR, exist_ok=True)
os.makedirs(UPDATES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

device = torch.device(CONFIG_DEVICE if torch.cuda.is_available() and CONFIG_DEVICE == 'cuda' else 'cpu')
print(f"Training API running on device: {device}")

MAX_LOGS = 5000
training_state = {
    'is_training': False,
    'current_round': 0,
    'current_client': 0,
    'current_epoch': 0,
    'total_rounds': NUM_ROUNDS,
    'total_clients': NUM_CLIENTS,
    'logs': [],
    'aggregation_logs': [],
    'error': None
}

training_lock = threading.Lock()
stop_event = threading.Event()


# ------------------- Helpers ------------------- #
def safe_log(log_type, message, round_num=0, client_id=0, epoch=0, metrics=None):
    entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'type': log_type,
        'message': message,
        'round': round_num,
        'client': client_id,
        'epoch': epoch,
        'metrics': metrics or {}
    }
    training_state['logs'].append(entry)
    if len(training_state['logs']) > MAX_LOGS:
        training_state['logs'] = training_state['logs'][-MAX_LOGS:]
    print(f"[{log_type.upper()}] {message}")


def atomic_save(obj, path, max_retries=3, sleep_s=0.2):
    tmp_path = path + ".tmp"
    last_exc = None
    for attempt in range(max_retries):
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
            return True
        except Exception as e:
            last_exc = e
            time.sleep(sleep_s)
    try:
        torch.save(obj, path)
        return True
    except Exception as e:
        safe_log('error', f'Failed to save {path}: {str(e)}')
        raise last_exc or e


def load_state_with_retry(path, max_retries=6, sleep_s=0.3):
    last_exc = None
    for attempt in range(max_retries):
        try:
            state = torch.load(path, map_location=device)
            return state
        except Exception as e:
            last_exc = e
            time.sleep(sleep_s)
    raise last_exc


def ensure_updates_clean():
    try:
        for f in os.listdir(UPDATES_DIR):
            p = os.path.join(UPDATES_DIR, f)
            try:
                os.remove(p)
            except Exception:
                pass
    except FileNotFoundError:
        os.makedirs(UPDATES_DIR, exist_ok=True)


def init_client_log(client_id):
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"client_{client_id}_log.csv")
    if not os.path.exists(log_path):
        try:
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "round", "epoch", "train_loss", "val_loss", "val_dice"])
        except Exception as e:
            safe_log('error', f'Failed to create client log {log_path}: {e}', 0, client_id)
    return log_path


def append_client_log(log_path, round_num, epoch, train_loss, val_loss, val_dice):
    try:
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round_num,
                epoch,
                f"{train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{val_dice:.4f}",
            ])
    except Exception as e:
        safe_log('error', f'Failed to append client log {log_path}: {e}')


if isinstance(TRAIN_SIZES, dict):
    sizes_list = [TRAIN_SIZES.get(i, 1) for i in range(1, NUM_CLIENTS + 1)]
else:
    sizes_list = list(TRAIN_SIZES)


# ------------------- Federated Training Class ------------------- #
class AutomatedFederatedTraining:
    def __init__(self):
        self.device = device
        self.global_model = get_unetr(self.device)
        self.shared_dir = SHARED_DIR
        self.global_path = os.path.join(GLOBAL_DIR, "global_latest.pth")

    def _add_log(self, log_type, message, round_num=0, client_id=0, epoch=0, metrics=None):
        safe_log(log_type, message, round_num, client_id, epoch, metrics)

    def initialize_global_model(self):
        if not os.path.exists(self.global_path):
            atomic_save(self.global_model.state_dict(), self.global_path)
            self._add_log('info', 'Initialized global model', 0, 0)
        else:
            try:
                state = torch.load(self.global_path, map_location=self.device)
                self.global_model.load_state_dict(state)
                self._add_log('info', 'Loaded existing global model', 0, 0)
            except Exception as e:
                self._add_log('error', f'Failed to load global model: {e}', 0, 0)
                raise

    def train_single_client(self, client_id, round_num):
        if stop_event.is_set():
            self._add_log('info', f'Stop requested before client {client_id} training', round_num, client_id)
            return False

        try:
            self._add_log('info', f'Client {client_id} starting training', round_num, client_id)
            train_loader, val_loader, _ = get_client_data(client_id)
            if train_loader is None:
                raise Exception(f"No data for client {client_id}")

            client_log_path = init_client_log(client_id)
            client_model = get_unetr(self.device)
            client_model.load_state_dict(load_state_with_retry(self.global_path))
            self._add_log('info', f'Client {client_id} loaded global model', round_num, client_id)

            optimizer = torch.optim.Adam(client_model.parameters(), lr=LEARNING_RATE)
            criterion = combined_loss

            for epoch in range(1, EPOCHS_PER_CLIENT + 1):
                if stop_event.is_set():
                    self._add_log('info', f'Stop requested during client {client_id} training', round_num, client_id, epoch)
                    return False

                training_state['current_epoch'] = epoch

                train_loss = train_one_epoch(client_model, train_loader, optimizer, criterion, self.device)
                val_loss, val_dice = evaluate(client_model, val_loader, criterion, self.device)

                metrics = {
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                    'val_dice': float(val_dice),
                    'val_dice_percentage': round(float(val_dice) * 100, 2)
                }

                append_client_log(client_log_path, round_num, epoch, train_loss, val_loss, val_dice)

                # Only log metrics for actual epoch training
                self._add_log(
                    'train',
                    f'Client {client_id} Epoch {epoch}/{EPOCHS_PER_CLIENT} - Loss: {train_loss:.4f}, Val Dice: {val_dice:.4f}',
                    round_num, client_id, epoch, metrics
                )

            update_path = os.path.join(UPDATES_DIR, f'client_{client_id}_weights.pth')
            atomic_save(client_model.state_dict(), update_path)
            self._add_log('info', f'Client {client_id} uploaded weights', round_num, client_id)

            return True

        except Exception as e:
            self._add_log('error', f'Client {client_id} training failed: {str(e)}', round_num, client_id)
            traceback.print_exc()
            raise

    def aggregate_models(self, round_num, wait_for_all=True, wait_timeout=60):
        try:
            self._add_log('info', f'Server starting aggregation for Round {round_num}', round_num, 0)
            start_wait = time.time()

            while True:
                client_files = [
                    os.path.join(UPDATES_DIR, f'client_{cid}_weights.pth')
                    for cid in range(1, NUM_CLIENTS + 1)
                    if os.path.exists(os.path.join(UPDATES_DIR, f'client_{cid}_weights.pth'))
                ]

                if not wait_for_all or len(client_files) == NUM_CLIENTS:
                    break

                if stop_event.is_set():
                    self._add_log('info', 'Stop requested during wait-for-updates', round_num, 0)
                    return False

                if time.time() - start_wait > wait_timeout:
                    self._add_log('error', f'Timeout waiting for client updates: found {len(client_files)}/{NUM_CLIENTS}', round_num, 0)
                    raise Exception(f'Expected {NUM_CLIENTS} client updates, found {len(client_files)}')
                time.sleep(1.0)

            state_dicts = [load_state_with_retry(f) for f in sorted(client_files)]
            if len(state_dicts) != NUM_CLIENTS:
                raise Exception(f'Expected {NUM_CLIENTS} client updates, found {len(state_dicts)}')

            new_state = fed_avg(state_dicts, sizes_list)
            self.global_model.load_state_dict(new_state)

            atomic_save(new_state, self.global_path)
            round_path = os.path.join(GLOBAL_DIR, f'global_round_{round_num}.pth')
            atomic_save(new_state, round_path)

            self._add_log('aggregation', f'Round {round_num} aggregation complete - Global model saved', round_num, 0)

            for f in client_files:
                try: os.remove(f)
                except Exception: pass

            training_state['aggregation_logs'].append({
                'round': round_num,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'num_clients': NUM_CLIENTS,
                'model_path': round_path
            })

            return True

        except Exception as e:
            self._add_log('error', f'Aggregation failed: {str(e)}', round_num, 0)
            traceback.print_exc()
            raise

    def run_federated_training(self):
        try:
            stop_event.clear()
            training_state['is_training'] = True
            training_state['error'] = None
            training_state['logs'] = []
            training_state['aggregation_logs'] = []
            training_state['current_round'] = 0
            training_state['current_client'] = 0
            training_state['current_epoch'] = 0

            self.initialize_global_model()

            for round_num in range(START_ROUND, NUM_ROUNDS + 1):
                if stop_event.is_set():
                    self._add_log('info', 'Stop requested before starting next round', round_num, 0)
                    break

                training_state['current_round'] = round_num
                training_state['current_client'] = 0
                training_state['current_epoch'] = 0

                self._add_log('info', f'===== ROUND {round_num}/{NUM_ROUNDS} STARTED =====', round_num, 0)

                ensure_updates_clean()

                for client_id in range(1, NUM_CLIENTS + 1):
                    if stop_event.is_set(): break
                    training_state['current_client'] = client_id
                    self.train_single_client(client_id, round_num)

                if stop_event.is_set(): break

                self.aggregate_models(round_num)
                self._add_log('info', f'===== ROUND {round_num}/{NUM_ROUNDS} COMPLETE =====', round_num, 0)

            if not stop_event.is_set():
                self._add_log('info', f'🎉 Federated training completed! All {NUM_ROUNDS} rounds finished.', NUM_ROUNDS, 0)
            else:
                self._add_log('info', 'Federated training stopped by user.', training_state['current_round'], training_state['current_client'])

        except Exception as e:
            training_state['error'] = str(e)
            self._add_log('error', f'Training failed: {str(e)}', training_state['current_round'], training_state['current_client'])
            traceback.print_exc()
        finally:
            training_state['is_training'] = False
            training_state['current_client'] = 0
            training_state['current_epoch'] = 0


# ------------------- API Endpoints ------------------- #
@app.route('/api/training/start', methods=['POST'])
def start_training():
    with training_lock:
        if training_state['is_training']:
            return jsonify({'success': False, 'error': 'Training already in progress'}), 400

        stop_event.clear()
        trainer = AutomatedFederatedTraining()
        thread = threading.Thread(target=trainer.run_federated_training)
        thread.daemon = False
        thread.start()

        return jsonify({
            'success': True,
            'message': 'Automated federated training started',
            'config': {
                'num_rounds': NUM_ROUNDS,
                'num_clients': NUM_CLIENTS,
                'epochs_per_client': EPOCHS_PER_CLIENT,
                'device': str(device),
                'start_round': START_ROUND
            }
        })


@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    stop_event.set()
    return jsonify({'success': True, 'message': 'Stop requested'})


@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    return jsonify({
        'is_training': training_state['is_training'],
        'current_round': training_state['current_round'],
        'current_client': training_state['current_client'],
        'current_epoch': training_state['current_epoch'],
        'total_rounds': training_state['total_rounds'],
        'total_clients': training_state['total_clients'],
        'error': training_state['error'],
        'progress_percentage': calculate_progress()
    })


@app.route('/api/training/logs', methods=['GET'])
def get_training_logs():
    limit = request.args.get('limit', default=100, type=int)
    return jsonify({'logs': training_state['logs'][-limit:], 'total_logs': len(training_state['logs'])})


@app.route('/api/training/aggregations', methods=['GET'])
def get_aggregation_logs():
    return jsonify({'aggregations': training_state['aggregation_logs']})


@app.route('/api/training/metrics', methods=['GET'])
def get_training_metrics():
    """
    Return metrics for each client:
    - Uses only logs of type 'train'
    - Aggregates by actual client_id
    - Avoid duplicates
    """
    metrics_by_client = {}

    for log in training_state['logs']:
        if log['type'] == 'train' and log.get('metrics') and log['client'] > 0:
            client_id = log['client']
            round_num = log['round']

            if client_id not in metrics_by_client:
                metrics_by_client[client_id] = []

            metrics_by_client[client_id].append({
                'round': round_num,
                'epoch': log['epoch'],
                'train_loss': log['metrics']['train_loss'],
                'val_loss': log['metrics']['val_loss'],
                'val_dice': log['metrics']['val_dice'],
                'val_dice_percentage': log['metrics']['val_dice_percentage']
            })

    return jsonify({'metrics_by_client': metrics_by_client})


@app.route('/api/training/reset', methods=['POST'])
def reset_training():
    with training_lock:
        if training_state['is_training']:
            return jsonify({'success': False, 'error': 'Cannot reset while training is in progress'}), 400

        training_state.update({
            'current_round': 0,
            'current_client': 0,
            'current_epoch': 0,
            'logs': [],
            'aggregation_logs': [],
            'error': None
        })
        ensure_updates_clean()
        return jsonify({'success': True, 'message': 'Training state reset'})


def calculate_progress():
    if not training_state['is_training']:
        return 0
    try:
        total_steps = NUM_ROUNDS * NUM_CLIENTS * EPOCHS_PER_CLIENT
        completed_steps = ((training_state['current_round'] - 1) * NUM_CLIENTS * EPOCHS_PER_CLIENT +
                           (training_state['current_client'] - 1) * EPOCHS_PER_CLIENT +
                           training_state['current_epoch'])
        return round((completed_steps / total_steps) * 100, 2)
    except Exception:
        return 0


if __name__ == '__main__':
    print(f"Training API starting on {device} ...")
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
