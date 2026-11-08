import copy
import sys
from pathlib import Path

from industrial_ad import (
    DEFAULT_QUANTIZATION_CONFIG,
    clone_config,
    discover_run_dirs,
    run_quantizations,
)
from industrial_ad.utils import load_json


##### base config #####

families = [
    "MLP-reconstruction",
    "TCN-reconstruction",
    "GRU-repeated-reconstruction",
    "GRU-seq2seq-reconstruction",
    "Transformer-reconstruction",
    "Conv-reconstruction",
    "MLP-forecasting",
    "TCN-forecasting",
    "GRU-seq2seq-forecasting",
]

base_config = clone_config(DEFAULT_QUANTIZATION_CONFIG)
base_config["run"] = {
    "name": None,
    "dir": "./runs/",
    "seed": None,
    "notes": "Post-training quantization of a trained detector.",
    "tags": ["quantized", "ptq"],
}   
base_config["source"] = {
    "run_dir": None,
    "checkpoint": "best",
}
base_config["quantization"] = {
    "backend": "fbgemm",
    "dtype": "qint8",
    "calibration_batches": None,
}
base_config["benchmark"]["enabled"] = False


##### source runs prepare #####


sweep_configs = []
for family in families:
    for source_run_dir in discover_run_dirs(Path("runs") / family):
        source_config = load_json(source_run_dir / "config.json")
        config = copy.deepcopy(base_config)
        target_family_dir = source_run_dir.parent.with_name(f"{source_run_dir.parent.name}-quant")

        config["run"] = {
            **config["run"],
            "name": source_run_dir.name,
            "dir": str(target_family_dir / source_run_dir.name),
            "seed": source_config["run"]["seed"],
            "tags": [*source_config["run"].get("tags", []), *config["run"]["tags"]],
        }
        config["source"] = {
            **config["source"],
            "run_dir": str(source_run_dir),
        }
        sweep_configs.append(config)

print("total planned quantized runs:", len(sweep_configs))

##### slice source runs #####

args = list(map(lambda x: None if x == "None" else int(x), sys.argv[1:]))
if not args:
    args = [None, None, None]
sweep_configs = sweep_configs[slice(*args)]
print("planned quantized runs after slice:", len(sweep_configs))

##### run quantization #####

run_quantizations(sweep_configs, skip_existing=True)