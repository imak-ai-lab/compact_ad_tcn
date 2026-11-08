import copy
from pathlib import Path

from industrial_ad import (
    DEFAULT_EXPERIMENT_CONFIG,
    clone_config,
    run_experiments,
)

##### base config #####

base_name = 'Conv-reconstruction'
base_config = clone_config(DEFAULT_EXPERIMENT_CONFIG)
base_config["run"] = {
    "name": None,
    "dir": f"./runs/{base_name}/",
    "seed": None,
    "notes": "Conv reconstruction for PU.",
    "tags": ["conv", "reconstruction", "pu"],
}
base_config["task"] = {
    "type": "reconstruction",
}
base_config["model"] = {
    "name": "conv_ae",
    "params": {
        "activation": "relu",

        "hidden_channels": None, 
        "latent_channels": None,
        "kernel_size": None,
    },
}
base_config["optimizer"] = {
    "name": "adamw",
    "params": {
        "lr": 1e-3,
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
    "epochs": 160,
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

all_hidden_sizes = [32, 48, 64, 96, 128]
hidden_variants = []
for mask in range(1, 2 ** len(all_hidden_sizes)):
    cur = []
    inds = []
    for bit in range(len(all_hidden_sizes)):
        if (mask >> bit) & 1:
            inds.append(bit)
            cur.append(all_hidden_sizes[bit])
    if 32 not in cur:
        continue
    diffs = [inds[i] - inds[i-1] for i in range(1, len(cur))]
    if len(diffs) > 0 and min(diffs) != max(diffs):
        continue
    hidden_variants.append(cur)


sweep_configs = []
cur_id = 0
for hidden_channels in hidden_variants:
    if 128 // 2 ** len(hidden_channels) == 0:
        raise ValueError("Too many hidden layers, the time dimension is reduced to 0.")
    last = hidden_channels[-1]
    for latent_channels in [last // 8, last // 4, last // 2]:
        input_elements = 27 * 120
        latent_elements = (120 // 2 ** len(hidden_channels)) * latent_channels
        if not (input_elements // 20 <= latent_elements <= input_elements // 4):
            continue
        if latent_channels == 0:
            continue
        for kernel_size in [3, 5, 7, 9]:
            cur_id += 1
            for seed in seeds:
                config = copy.deepcopy(base_config)
                
                name = f"{cur_id:0>3}-s{seed}-hid{'.'.join(map(str, hidden_channels))}-lat{latent_channels}-ker{kernel_size}"
                config["run"] = {
                    **config["run"],
                    "name": name,
                    "dir": str(Path(config["run"]["dir"]) / name),
                    "seed": seed,
                }
                config["model"]["params"]["hidden_channels"] = hidden_channels
                config["model"]["params"]["latent_channels"] = latent_channels
                config["model"]["params"]["kernel_size"] = kernel_size


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
