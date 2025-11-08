# General FL parameters
NUM_CLIENTS = 4
NUM_ROUNDS = 10
EPOCHS_PER_CLIENT = 3
BATCH_SIZE = 1
START_ROUND = 1
LEARNING_RATE = 1e-4

# Dataset sizes for each client (train, val, test)
# Used for weighted FedAvg aggregation (based on training size)
CLIENT_DATA_SIZES = {
    1: {"train": 56, "val": 13, "test": 12},
    2: {"train": 56, "val": 13, "test": 12},
    3: {"train": 56, "val": 13, "test": 12},
    4: {"train": 56, "val": 13, "test": 12},
}

# Derived values for aggregation
TRAIN_SIZES = [CLIENT_DATA_SIZES[c]["train"] for c in sorted(CLIENT_DATA_SIZES.keys())]

# Paths
BASE_DIR = "./"
SHARED_DIR = "shared"  # for model exchange between clients and server

# Hardware
DEVICE = "cuda"  # or "cpu"
