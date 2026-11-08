import copy
from pathlib import Path

from industrial_ad import (
    DEFAULT_EXPERIMENT_CONFIG,
    clone_config,
    run_experiments,
)

##### base config #####

base_name = "GRU-seq2seq-forecasting"
base_config = clone_config(DEFAULT_EXPERIMENT_CONFIG)
base_config["run"] = {
    "name": None,
    "dir": f"./runs/{base_name}/",
    "seed": None,
    "notes": "GRU seq2seq forecasting for PU.",
    "tags": ["gru", "forecasting", "pu"],
}
base_config["task"] = {
    "type": "forecasting",
}
base_config["model"] = {
    "name": "gru_seq2seq_forecaster",
    "params": {
        "dropout": 0.0,

        "hidden_size": None,
        "num_layers": None,
        "teacher_forcing_ratio": None,
    },
}
base_config["optimizer"] = {
    "name": "adamw",
    "params": {
        "lr": 3e-4,
        "weight_decay": 1e-4,
    },
}
base_config["scheduler"] =  {
    "name": "warmup_cosine",
    "params": {
        "warmup_epochs": 10,
        "min_lr": 1e-6,
        "start_factor": 1e-4,
    },
}
base_config["trainer"] = {
    "epochs": 110,
    "eval_every_n_epochs": 10,
    "device": "cuda",
    "amp": False,
    "clip_grad_norm": 1.0,
    "checkpoint": {
        "metric": "val/roc_auc",
        "mode": "max",
        "min_delta": 0.0,
    },
    "early_stopping": {
        "enabled": False,
        "metric": "val/roc_auc",
        "mode": "max",
        "patience": 12,
        "min_delta": 0.0,
    },
}

##### hyperparameters prepare #####

seeds = [42, 43, 44]

sweep_configs = []
cur_id = 0

for horizon_size in [16, 32, 64, 120]:
    for teacher_forcing_ratio in [0.25]:
        for num_layers in [1,2,3]:
            for hidden_size in [32, 48, 64, 96, 128, 192]:
                if num_layers == 2 and hidden_size >= 192:
                    continue
                if num_layers == 3 and hidden_size >= 128:
                    continue
                cur_id += 1
                for seed in seeds:
                    config = copy.deepcopy(base_config)
                    name = f"{cur_id:0>3}-s{seed}-hor{horizon_size}-lay{num_layers}-hid{hidden_size}-rev0-teach{teacher_forcing_ratio:.2}"
                    config["run"] = {
                        **config["run"],
                        "name": name,
                        "dir": str(Path(config["run"]["dir"]) / name),
                        "seed": seed,
                    }
                    config["model"]["params"]["hidden_size"] = hidden_size
                    config["model"]["params"]["num_layers"] = num_layers
                    config["model"]["params"]["teacher_forcing_ratio"] = teacher_forcing_ratio
                    config["dataset"]["params"]["horizon_size"] = horizon_size

                    sweep_configs.append(config)

print("total planned runs:", len(sweep_configs))

##### slice hyperparameters #####

import sys
args = list(map(lambda x: None if x == "None" else int(x), sys.argv[1:]))
if not args: args = [None, None, None]
sweep_configs = sweep_configs[slice(*args)]
print("planned runs after slice:", len(sweep_configs))

##### run experiments #####

run_experiments(sweep_configs, skip_existing=True, dry_run=False)
