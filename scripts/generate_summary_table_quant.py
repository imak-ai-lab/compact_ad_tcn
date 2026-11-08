from industrial_ad import discover_run_dirs, load_run_dataframe

runs = discover_run_dirs(
    "runs/MLP-reconstruction-quant", 
    "runs/TCN-reconstruction-quant", 
    "runs/Conv-reconstruction-quant", 
    "runs/GRU-repeated-reconstruction-quant", 
    "runs/GRU-seq2seq-reconstruction-quant",
    'runs/Transformer-reconstruction-quant',
    "runs/MLP-forecasting-quant", 
    "runs/TCN-forecasting-quant",
    "runs/TCN-light-forecasting-quant",
    "runs/GRU-seq2seq-forecasting-quant",
)
df = load_run_dataframe(runs)
df["seed"] = df["run_name"].apply(lambda x: int(x.split('-')[1][1:]))
df["run_name"] = df["run_name"].apply(lambda x: '-'.join(x.split('-')[:1] + x.split('-')[2:]))
smaller_df = df[
    [
        "family", "run_name", "seed",
        *[c for c in df.columns if c.startswith("best_metrics.") if "confusion_matrix" not in c and "threshold" not in c and '.train/' not in c],
        "parameter_size_bytes",
        "benchmark.latency_mean_seconds", "benchmark.latency_std_seconds", "benchmark.peak_memory_bytes",
    ]
]

float64_cols = list(smaller_df.select_dtypes(include='float64'))
smaller_df[float64_cols] = smaller_df[float64_cols].astype('float32')

smaller_df.to_csv("summary_table_quant.csv", index=False)
