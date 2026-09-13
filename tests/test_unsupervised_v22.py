import numpy as np
import pandas as pd

from research_bot.unsupervised_v22 import RegimeAnomalyFeatureLab, UnsupervisedConfig, tail_enrichment


def dataset(n=700, seed=314):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    return pd.DataFrame({
        "signal_time": t,
        "f_ret1": x,
        "f_atr_pct": np.abs(y) * 0.01,
        "f_trend": 0.5 * x + 0.2 * rng.normal(size=n),
        "label_r_multiple": np.where(x < -1.7, -2.0, rng.normal(0, 0.5, n)),
    })


def test_unsupervised_fit_transform_has_no_label_inputs():
    df = dataset()
    lab = RegimeAnomalyFeatureLab(UnsupervisedConfig(n_components=3, n_clusters=4, max_fit_rows=500)).fit(df.iloc[:500])
    assert all(c.startswith("f_") for c in lab.columns)
    assert "label_r_multiple" not in lab.columns
    z = lab.transform(df.iloc[500:])
    assert len(z) == 200
    assert "u_kmeans_regime" in z
    assert "u_isolation_score" in z


def test_future_label_mutation_cannot_change_transformed_features():
    df = dataset()
    train = df.iloc[:500].copy()
    hold = df.iloc[500:].copy()
    lab = RegimeAnomalyFeatureLab(UnsupervisedConfig(n_components=3, n_clusters=4)).fit(train)
    a = lab.transform(hold)
    hold["label_r_multiple"] = 999.0
    b = lab.transform(hold)
    assert np.allclose(a.select_dtypes(include=[np.number]), b.select_dtypes(include=[np.number]), equal_nan=True)


def test_tail_enrichment_is_descriptive_and_finite_when_possible():
    df = dataset()
    df["u_test_anomaly"] = (df["f_ret1"] < -1.5).astype(int)
    m = tail_enrichment(df, "u_test_anomaly")
    assert m["rows"] == len(df)
    assert 0 <= m["anomaly_rate"] <= 1
    assert m["tail_rate_anomaly"] >= m["tail_rate_all"]
