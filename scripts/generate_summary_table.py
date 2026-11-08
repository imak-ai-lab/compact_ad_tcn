from industrial_ad import discover_run_dirs, load_run_dataframe

runs = discover_run_dirs(
    "runs/MLP-reconstruction", 
    "runs/TCN-reconstruction", 
    "runs/Conv-reconstruction", 
    "runs/GRU-repeated-reconstruction", 
    "runs/GRU-seq2seq-reconstruction",
    'runs/Transformer-reconstruction',
    "runs/PCA-reconstruction",
    "runs/MLP-forecasting", 
    "runs/TCN-forecasting",
    "runs/TCN-light-forecasting",
    "runs/GRU-seq2seq-forecasting",
)
df = load_run_dataframe(runs)
df["seed"] = df["run_name"].apply(lambda x: int(x.split('-')[1][1:]))
df["run_name"] = df["run_name"].apply(lambda x: '-'.join(x.split('-')[:1] + x.split('-')[2:]))
smaller_df = df[
    [
        "family", "run_name", "seed", "best_epoch", "config.trainer.epochs",
        *[c for c in df.columns if c.startswith("best_metrics.") if "confusion_matrix" not in c and "threshold" not in c and '.train/' not in c],
        "best_metrics.train/loss", "last_metrics.train/loss",
        "parameter_count",
        "benchmark.latency_mean_seconds", "benchmark.latency_std_seconds", "benchmark.peak_memory_bytes",
    ]
].rename(columns={"config.trainer.epochs": "total_epochs"})

float64_cols = list(smaller_df.select_dtypes(include='float64'))
smaller_df[float64_cols] = smaller_df[float64_cols].astype('float32')

smaller_df.to_csv("summary_table.csv", index=False)
