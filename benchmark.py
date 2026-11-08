from industrial_ad import (
    benchmark_runs,
    discover_run_dirs,
)


#run_dirs = discover_run_dirs("runs/MLP-reconstruction-quant", "runs/TCN-reconstruction-quant", "runs/GRU-repeated-reconstruction-quant", "runs/GRU-seq2seq-reconstruction-quant", 'runs/Transformer-reconstruction-quant', 'runs/Conv-reconstruction-quant')
#run_dirs += discover_run_dirs("runs/PCA-reconstruction")
#run_dirs += discover_run_dirs("runs/MLP-forecasting-quant", "runs/TCN-forecasting-quant", "runs/GRU-seq2seq-forecasting-quant", "runs/TCN-light-forecasting")
run_dirs = discover_run_dirs("runs/TCN-light-forecasting-quant")




benchmark_runs(
    run_dirs=run_dirs,
    checkpoint="best",
    benchmark_config={
        "enabled": True,
        "device": "cpu",
        "num_threads": 1,
        "warmup_runs": 50,
        "num_runs": 2000,
        "profile_memory": True,
    },
    skip_existing=True
)
