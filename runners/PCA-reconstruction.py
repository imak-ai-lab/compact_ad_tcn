import copy
from pathlib import Path

from industrial_ad import (
    DEFAULT_EXPERIMENT_CONFIG,
    clone_config,
    run_experiments,
)

##### base config #####

base_name = "PCA-reconstruction"
base_config = clone_config(DEFAULT_EXPERIMENT_CONFIG)
base_config["run"] = {
    "name": None,
    "dir": f"./runs/{base_name}/",
    "seed": None,
    "notes": "PCA reconstruction baseline for PU.",
    "tags": ["pca", "reconstruction", "pu"],
}
base_config["task"] = {
    "type": "reconstruction",
}
base_config["model"] = {
    "name": "pca",
    "params": {
        "n_components": None,
        "svd_solver": "auto",
    },
}
base_config["loss"] = {
    "name": None,
    "params": {},
}
base_config["optimizer"] = {
    "name": None,
    "params": {},
}
base_config["scheduler"] = {
    "name": None,
    "params": {},
}
base_config["trainer"] = {
    "epochs": 1,
    "eval_every_n_epochs": 1,
    "device": "cpu",
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

n_component_variants = [2, 4, 8, 16, 32, 64, 128, 96]
seeds = [42, 43, 44]

sweep_configs = []
for run_id, n_components in enumerate(n_component_variants, start=1):
    for seed in seeds:
        config = copy.deepcopy(base_config)
        name = f"{run_id:0>3}-s{seed}-k{n_components}"
        config["run"] = {
            **config["run"],
            "name": name,
            "dir": str(Path(config["run"]["dir"]) / name),
            "seed": seed,
        }
        config["model"]["params"]["n_components"] = n_components
        sweep_configs.append(config)

print("total planned runs:", len(sweep_configs))

##### slice hyperparameters #####

import sys
args = list(map(lambda x: None if x == "None" else int(x), sys.argv[1:]))
if not args:
    args = [None, None, None]
sweep_configs = sweep_configs[slice(*args)]
print("planned runs after slice:", len(sweep_configs))

##### run experiments #####

run_experiments(sweep_configs, skip_existing=True, dry_run=False)
