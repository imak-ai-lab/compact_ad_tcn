import copy
from pathlib import Path

from industrial_ad import DEFAULT_EXPERIMENT_CONFIG, clone_config, run_experiments


base_name = 'MLP-forecasting'
base_config = clone_config(DEFAULT_EXPERIMENT_CONFIG)
base_config["run"] = {
    "name": None,
    "dir": f"./runs/{base_name}/",
    "seed": None,
    "notes": "MLP forecasting for PU.",
    "tags": ["mlp", "forecasting", "pu"],
}
base_config["task"] = {
    "type": "forecasting",
}
base_config["model"] = {
    "name": "mlp_forecaster",
    "params": {
        "activation": "relu",
        "hidden_dims": None,
        "dropout": 0.0,
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
    "epochs": 210,
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

input_sz = 27 * 120
all_hidden_sizes = [2**11, 2**10, 2**9, 2**8, 2**7, 2**6]
hidden_variants = []

for mask in range(1, 2 ** len(all_hidden_sizes)):
    cur = []
    inds = []
    for bit in range(len(all_hidden_sizes)):
        if (mask >> bit) & 1:
            inds.append(bit)
            cur.append(all_hidden_sizes[bit])
    diffs = [inds[i] - inds[i-1] for i in range(1, len(cur))]
    if len(diffs) > 0 and min(diffs) != max(diffs):
        continue
    cur = cur + cur[::-1]
    for i in range(1, len(cur) + 1):
        if cur[:i] in hidden_variants: continue
        hidden_variants.append(cur[:i])

def calculate_number_of_parameters(input_dim, output_dim, hidden_dims):
    total_params = 0
    prev_dim = input_dim
    for hidden_dim in hidden_dims + [output_dim]:
        total_params += prev_dim * hidden_dim + hidden_dim  # weights + biases
        prev_dim = hidden_dim
    return total_params

seeds = [42, 43, 44]

sweep_configs = []
cur_id = 0
for horizon_size in [8, 16, 32, 64]:
    for hidden_dims in hidden_variants:
        if hidden_dims[-1] <= horizon_size * 27:
            continue
        if len(hidden_dims) > 8:
            continue

        # approximate size of all matrices
        params = calculate_number_of_parameters(120*27, horizon_size*27, hidden_dims)
        if params > 12e6 or (horizon_size < 64 and params > 9e6):
            continue


        cur_id += 1
        for seed in seeds:
            config = copy.deepcopy(base_config)
            name = f"{cur_id:0>3}-s{seed}-h{horizon_size}-hid{'.'.join(map(str, hidden_dims))}"
            config["run"] = {
                **config["run"],
                "name": name,
                "dir": str(Path(config["run"]["dir"]) / name),
                "seed": seed,
            }
            config["dataset"]["params"]["horizon_size"] = horizon_size
            config["model"]["params"]["hidden_dims"] = hidden_dims
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

