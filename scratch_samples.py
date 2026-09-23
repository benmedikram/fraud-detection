import pandas as pd

test = pd.read_csv("data/processed/test.csv")

# 3 normal + 3 fraud examples, with only the raw columns the API expects
raw_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
normal_samples = test[test["Class"] == 0].sample(3, random_state=1)[raw_cols]
fraud_samples = test[test["Class"] == 1].sample(3, random_state=1)[raw_cols]

samples = pd.concat([normal_samples, fraud_samples])
samples.to_csv("data/processed/api_samples.csv", index=False)
print(samples)