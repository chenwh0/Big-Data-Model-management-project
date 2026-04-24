import pandas as pd
import numpy as np
import ray
import xgboost as xgb
import matplotlib.pyplot as plt
from ray.data import from_pandas
from time import time
from sklearn.metrics import mean_squared_error

# ======================
# CONFIG
# ======================
CHARACTERISTIC_NAME = "pH"
TARGET = "ResultMeasureValue"
DROP_COLUMNS = [TARGET] #, "seasonID", "season", "medianYearValue", "minYearValue", "maxYearValue", "avgYearValue", "logAvgYearValue", "medianSeasonValue", "minSeasonValue", "maxSeasonValue", "avgSeasonValue"]
DATA_PATH = f"/home/ubuntu/project_phase2/seasonal/{CHARACTERISTIC_NAME}/"
OUTPUT_DIR = "/home/ubuntu/project_phase2/actual_vs_predict_visualizations/no_temporal/"

# ======================
# INIT RAY (local GPU)
# ======================
ray.init(ignore_reinit_error=True)

# ======================
# LOAD DATA
# ======================
df = pd.read_parquet(DATA_PATH)

print("Original shape:", df.shape)

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])

y = df[TARGET]
X = df.drop(columns=DROP_COLUMNS)

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.fillna(X.median(numeric_only=True))

df_clean = X.copy()
df_clean[TARGET] = y
print("num columns:", len(df_clean.columns))
# ======================
# RAY DATA SPLIT
# ======================
ds = from_pandas(df_clean)
split = ds.train_test_split(test_size=0.2, seed=42)
train_ds = split[0]
eval_ds = split[1]
print("total rows:", ds.count())
print("train rows:", train_ds.count())
print("eval rows:", eval_ds.count())
# ======================
# TRAIN FUNCTION (NO RABIT / NO DISTRIBUTED XGB)
# ======================
@ray.remote(num_gpus=1)
def train_xgb(train_df, eval_df):

    train_df = train_df.to_pandas()
    eval_df = eval_df.to_pandas()

    y_train = train_df[TARGET]
    y_eval = eval_df[TARGET]

    X_train = train_df.drop(columns=[TARGET]).astype(float)
    X_eval = eval_df.drop(columns=[TARGET]).astype(float)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_eval, label=y_eval)

    params = {
        "tree_method": "hist",
        "device": "cuda",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 8,
        "eta": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 1.0,
        "nthread": 1
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(deval, "validation")],
        early_stopping_rounds=20,
        verbose_eval=10
    )

    bst.save_model("best_model.json")
    return bst

# ======================
# RUN TRAINING
# ======================
print("\nStarting training...\n")
start_time = time()
bst_ref = train_xgb.remote(train_ds, eval_ds)
bst = ray.get(bst_ref)
end_time = time()
print("\nTraining complete.")
print(f"Best train iteration {bst.best_iteration}")
print(f"Best train iteration score {bst.best_score}")
print(f"Training time: {end_time - start_time:.2f} seconds")
# ======================
# VISUALIZATION
# ======================
train_df = train_ds.to_pandas()
eval_df = eval_ds.to_pandas()
y_train = train_df[TARGET]
y_eval = eval_df[TARGET]
X_eval = eval_df.drop(columns=[TARGET]).astype(float)

dtest = xgb.DMatrix(X_eval)
y_pred = bst.predict(dtest)

rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
print("FINAL TEST RMSE:", rmse)

baseline_pred = np.full_like(y_eval, y_train.mean())
baseline_rmse = np.sqrt(mean_squared_error(y_eval, baseline_pred))

print("Baseline RMSE:", baseline_rmse)
print("Improvement over baseline:", (baseline_rmse - rmse) / baseline_rmse)
print(f"Training time: {end_time - start_time:.2f} seconds")
# ---- Actual vs Predicted ----
plt.figure()

max_value = max(np.max(y_eval), np.max(y_pred))
plt.xlim(0, max_value)
plt.ylim(0, max_value)
plt.plot([0, max_value], [0, max_value], "black")

plt.scatter(y_eval, y_pred, alpha=0.3)
plt.xlabel("Actual")
plt.ylabel("Predicted")

characteristic = CHARACTERISTIC_NAME.upper().replace("_", " ")
plt.title(f"{characteristic} Actual vs Predicted")

plt.savefig(f"{OUTPUT_DIR}{CHARACTERISTIC_NAME}_actual_vs_predicted.png")
plt.close()

# ---- Feature Importance ----
plt.figure()
xgb.plot_importance(bst, max_num_features=10)
plt.title("Feature Importance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}{CHARACTERISTIC_NAME}_feature_importance.png")
plt.close()

print("\nPlots saved successfully.")
