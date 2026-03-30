import pandas as pd
from ml.shared.features.encode import make_categorical
from ml.shared.features.feature_defs import CAT_COLS


def test_make_categorical():
    df = pd.DataFrame({c: ["a", "b"] for c in CAT_COLS})
    result = make_categorical(df, CAT_COLS)
    for c in CAT_COLS:
        assert result[c].dtype.name == "category"
