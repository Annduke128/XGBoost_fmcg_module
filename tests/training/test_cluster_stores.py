import pandas as pd

from ml.training.segmentation.cluster_stores import cluster_stores


def test_cluster_stores_basic():
    profiles = pd.DataFrame(
        {
            "branch_id": [f"b{i}" for i in range(10)],
            "avg_weekly_sales": [10, 12, 11, 13, 9, 30, 32, 31, 29, 28],
            "volatility": [0.1] * 10,
            "promo_lift": [1.1] * 10,
            "stockout_rate": [0.05] * 10,
            "seasonality_strength": [1.2] * 10,
            "store_type": ["A"] * 5 + ["B"] * 5,
            "display_capacity_type": ["S"] * 10,
            "service_scale": ["L"] * 10,
            "channel": ["offline"] * 10,
        }
    )

    clustered = cluster_stores(profiles, n_clusters=2, min_stores=2, random_state=0)
    assert "cluster" in clustered.columns
    assert clustered["cluster"].nunique() <= 2


def test_cluster_stores_min_stores_enforced():
    profiles = pd.DataFrame(
        {
            "branch_id": [f"b{i}" for i in range(6)],
            "avg_weekly_sales": [10, 11, 12, 100, 101, 102],
            "volatility": [0.1] * 6,
            "promo_lift": [1.0] * 6,
            "stockout_rate": [0.0] * 6,
            "seasonality_strength": [1.0] * 6,
            "store_type": ["A"] * 6,
            "display_capacity_type": ["S"] * 6,
            "service_scale": ["L"] * 6,
            "channel": ["offline"] * 6,
        }
    )

    clustered = cluster_stores(profiles, n_clusters=3, min_stores=4, random_state=0)
    assert clustered["cluster"].nunique() <= 1
