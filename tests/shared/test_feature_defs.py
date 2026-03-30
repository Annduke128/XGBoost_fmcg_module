from ml.shared.features.feature_defs import CAT_COLS, NUM_COLS, ALL_FEATURES


def test_all_features_is_union():
    assert set(ALL_FEATURES) == set(CAT_COLS) | set(NUM_COLS)


def test_no_duplicates():
    assert len(ALL_FEATURES) == len(set(ALL_FEATURES))
