import pandas as pd
import ray
import xgboost as xgb
import matplotlib.pyplot as plt

from ray.data import from_pandas
from ray.train.xgboost import XGBoostTrainer, RayTrainReportCallback
from ray.train import ScalingConfig


TARGET = "ResultMeasureValue"
DATA_PATH = "/home/ubuntu/phase2/split_parquet/Dissolved_oxygen.parquet"


ray.init(ignore_reinit_error=True)


df = pd.read_parquet(DATA_PATH)

print("Original shape:", df.shape)

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])

y = df[TARGET]

X = df.drop(columns=[TARGET])

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

#fill missing values with median
X = X.fillna(X.median(numeric_only=True))

df_clean = X.copy()
df_clean[TARGET] = y

ds = from_pandas(df_clean)

split = ds.train_test_split(test_size=0.2, seed=42)
train_ds = split[0]
eval_ds = split[1]

#training
def train_fn_per_worker(config):

    train_ds = ray.train.get_dataset_shard("train").materialize()
    eval_ds = ray.train.get_dataset_shard("validation").materialize()

    train_df = train_ds.to_pandas()
    eval_df = eval_ds.to_pandas()

    y_train = train_df[TARGET]
    y_eval = eval_df[TARGET]

    X_train = train_df.drop(columns=[TARGET])
    X_eval = eval_df.drop(columns=[TARGET])

    X_train = X_train.astype(float)
    X_eval = X_eval.astype(float)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_eval, label=y_eval)

    
    params = {
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "max_depth": 8,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 1.0,
        "eval_metric": "rmse"
    }

    #early stopping
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(deval, "validation")],
        early_stopping_rounds=20,
        callbacks=[RayTrainReportCallback()],
        verbose_eval=10
    )

    #save best model per worker
    bst.save_model("best_model.json")

    return bst

trainer = XGBoostTrainer(
    train_loop_per_worker=train_fn_per_worker,
    datasets={
        "train": train_ds,
        "validation": eval_ds
    },
    scaling_config=ScalingConfig(
        num_workers=2,
        use_gpu=False
    )
)

import time
start_time = time.time()

result = trainer.fit()
end_time = time.time()

bst = RayTrainReportCallback.get_model(result.checkpoint)

print("\nTraining complete.")

#visualization
eval_df = eval_ds.to_pandas()

y_true = eval_df[TARGET]
X_eval = eval_df.drop(columns=[TARGET]).astype(float)

dtest = xgb.DMatrix(X_eval)
y_pred = bst.predict(dtest)

plt.figure()
plt.scatter(y_true, y_pred, alpha=0.3)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.savefig("actual_vs_predicted.png")
plt.close()

#feature importance
plt.figure()
xgb.plot_importance(bst, max_num_features=10)
plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.close()

#save best model
bst.save_model("DO.json")

print("Best model saved.")
print(f"Time taken: {end_time - start_time} s")
