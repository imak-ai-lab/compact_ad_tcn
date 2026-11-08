import copy
from pathlib import Path

from industrial_ad import (
    DEFAULT_EXPERIMENT_CONFIG,
    clone_config,
    run_experiments,
)

##### base config #####

base_name = 'TCN-reconstruction'
base_config = clone_config(DEFAULT_EXPERIMENT_CONFIG)
base_config["run"] = {
    "name": None,
    "dir": f"./runs/{base_name}/",
    "seed": None,
    "notes": "TCN reconstruction for PU.",
    "tags": ["tcn", "reconstruction", "pu"],
}
base_config["task"] = {
    "type": "reconstruction",
}
base_config["model"] = {
    "name": "tcn_ae",
    "params": {
        "activation": "relu",
        "dropout": 0.0,

        "hidden_channels": None, 
        "latent_channels": None,
        "num_blocks": None,
        "kernel_size": None,
        "separable": None, # true, false
        "norm": "layer", # batch, layer, none

        "dilations": None # None for 2^i - classic
    },
}
base_config["optimizer"] = {
    "name": "adamw",
    "params": {
        "lr": 3e-3,
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
for separable in [True, False]:
    for hidden_channels in [16, 32, 48, 64]:
        for latent_channels in [2, 4, 8]:
            for num_blocks in [2, 3, 4, 5]:
                for kernel_size in [3, 5, 7]:
                    
                    dilations = [2**index for index in range(num_blocks)]
                    receptive_field = 1 + 4 * (kernel_size - 1) * sum(dilations)
                    if receptive_field > 250 or receptive_field < 70:
                        continue

                    cur_id += 1
                    for seed in seeds:
                        config = copy.deepcopy(base_config)
                        
                        name = f"{cur_id:0>3}-s{seed}-sep{int(separable)}-hid{hidden_channels}-lat{latent_channels}-bl{num_blocks}-ker{kernel_size}"
                        config["run"] = {
                            **config["run"],
                            "name": name,
                            "dir": str(Path(config["run"]["dir"]) / name),
                            "seed": seed,
                        }
                        config["model"]["params"]["separable"] = separable
                        config["model"]["params"]["hidden_channels"] = hidden_channels
                        config["model"]["params"]["latent_channels"] = latent_channels
                        config["model"]["params"]["num_blocks"] = num_blocks
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

run_experiments(sweep_configs, skip_existing=True)
