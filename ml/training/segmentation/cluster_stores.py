from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_PROFILE_COLS = [
    "avg_weekly_sales",
    "volatility",
    "promo_lift",
    "stockout_rate",
    "seasonality_strength",
]
CAT_PROFILE_COLS = [
    "store_type",
    "display_capacity_type",
    "service_scale",
    "channel",
]


def _prepare_features(profiles: pd.DataFrame) -> np.ndarray:
    num_data = profiles[NUM_PROFILE_COLS].astype(float).to_numpy()
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_data)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_data = profiles[CAT_PROFILE_COLS].astype(str)
    cat_encoded = ohe.fit_transform(cat_data)

    return np.hstack([num_scaled, cat_encoded])


def _merge_small_clusters(
    labels: np.ndarray,
    centers: np.ndarray,
    min_stores: int,
) -> np.ndarray:
    labels = labels.copy()
    unique, counts = np.unique(labels, return_counts=True)
    sizes = dict(zip(unique, counts))

    large_clusters = [cid for cid, size in sizes.items() if size >= min_stores]
    if not large_clusters:
        return labels

    for cid, size in sizes.items():
        if size >= min_stores:
            continue
        distances = np.linalg.norm(centers[large_clusters] - centers[cid], axis=1)
        nearest_large = large_clusters[int(np.argmin(distances))]
        labels[labels == cid] = nearest_large

    return labels


def cluster_stores(
    profiles: pd.DataFrame,
    n_clusters: int = 5,
    min_stores: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    if profiles.empty:
        return profiles.assign(cluster=pd.Series(dtype=int))

    max_clusters = max(1, len(profiles) // max(min_stores, 1))
    n_clusters = min(n_clusters, max_clusters)
    n_clusters = max(n_clusters, 1)

    features = _prepare_features(profiles)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)

    labels = _merge_small_clusters(labels, kmeans.cluster_centers_, min_stores)

    return profiles.assign(cluster=labels)
