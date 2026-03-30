from eda.dataset_builder import build_and_save_dataset

summary = build_and_save_dataset(
    scenario_count=100,   # start small (fast)
    output_dir="generated_data",
    scenario_seed=42,
    split_seed=42,
)

print("\nDataset Build Summary:")
for k, v in summary.items():
    print(f"{k}: {v}")