import ray
import numpy as np
from sklearn.cluster import DBSCAN

ray.init()

# Read dataset
data = ray.data.read_parquet("/home/ubuntu/project_phase2/01filtered_parquets/")

# Convert to full Pandas DataFrame (GLOBAL)
dataframe = data.to_pandas()

# DBSCAN params
kms_per_radian = 6371.0088
epsilon = 10 / kms_per_radian
min_samples = 10

# Prepare coordinates
coords = dataframe[[
    "ActivityLocation/LatitudeMeasure",
    "ActivityLocation/LongitudeMeasure"
]].astype(float).to_numpy()

coords_rad = np.radians(coords)

# Run DBSCAN ON FULL DATASET
dbscan_model = DBSCAN(eps=epsilon, min_samples=min_samples, metric="haversine")
dataframe["cluster"] = dbscan_model.fit_predict(coords_rad)

# Save back using Ray (optional)
ray.data.from_pandas(dataframe).write_parquet("/home/ubuntu/project_phase2/02dbscan_cluster_parquets/")
