import pandas as pd

from src.data import add_features, clean, split


def make_df(n=1000):
    df = pd.DataFrame({"Time": range(n), "Amount": [10.0] * n, "Class": [0] * n})
    df.loc[:19, "Class"] = 1      # 20 frauds = 2%
    df["V1"] = range(n)           # makes every row unique
    return df


def test_clean_removes_duplicates():
    df = make_df()
    df_with_dupes = pd.concat([df, df.iloc[:5]])   # add 5 exact copies
    assert len(clean(df_with_dupes)) == 1000


def test_add_features_creates_columns():
    out = add_features(make_df())
    assert {"log_amount", "hour"} <= set(out.columns)


def test_split_keeps_fraud_ratio():
    train, val, test = split(add_features(make_df()))
    assert abs(train["Class"].mean() - 0.02) < 0.005
    assert abs(val["Class"].mean() - 0.02) < 0.01
    assert abs(test["Class"].mean() - 0.02) < 0.01