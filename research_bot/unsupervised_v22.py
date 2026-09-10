from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM


@dataclass(frozen=True)
class UnsupervisedConfig:
    n_components: int = 8
    n_clusters: int = 8
    random_seed: int = 314
    contamination: float = 0.05
    max_fit_rows: int = 80000


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = sorted(c for c in df.columns if c.startswith("f_") and pd.api.types.is_numeric_dtype(df[c]))
    if not cols:
        raise ValueError("No numeric f_* columns available for unsupervised learning")
    return cols


def deterministic_time_sample(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(df) <= limit:
        return df
    idx = np.linspace(0, len(df) - 1, limit, dtype=int)
    return df.iloc[idx].copy()


class RegimeAnomalyFeatureLab:
    """Fit unsupervised representations on development only and transform later periods.

    All models are fit without outcome labels. Generated columns are representations,
    regimes and anomaly scores; they are not trading signals by themselves.
    """

    def __init__(self, config: UnsupervisedConfig | None = None):
        self.config = config or UnsupervisedConfig()
        self.columns: list[str] = []
        self.preprocess: Pipeline | None = None
        self.pca: PCA | None = None
        self.kmeans: KMeans | None = None
        self.gmm: GaussianMixture | None = None
        self.isolation: IsolationForest | None = None
        self.lof: LocalOutlierFactor | None = None
        self.ocsvm: OneClassSVM | None = None
        self.elliptic: EllipticEnvelope | None = None

    def fit(self, development: pd.DataFrame) -> "RegimeAnomalyFeatureLab":
        x = development.sort_values("signal_time") if "signal_time" in development else development.copy()
        x = deterministic_time_sample(x, self.config.max_fit_rows)
        self.columns = feature_columns(x)
        self.preprocess = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
        ])
        z = self.preprocess.fit_transform(x[self.columns])
        n_comp = max(2, min(self.config.n_components, z.shape[1], max(2, len(z) - 1)))
        self.pca = PCA(n_components=n_comp, whiten=False, random_state=self.config.random_seed).fit(z)
        p = self.pca.transform(z)
        k = max(2, min(self.config.n_clusters, len(p)))
        self.kmeans = KMeans(n_clusters=k, n_init=20, random_state=self.config.random_seed).fit(p)
        self.gmm = GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-5, random_state=self.config.random_seed).fit(p)
        self.isolation = IsolationForest(n_estimators=300, contamination=self.config.contamination, random_state=self.config.random_seed, n_jobs=2).fit(p)
        self.lof = LocalOutlierFactor(n_neighbors=min(35, max(5, len(p) // 20)), contamination=self.config.contamination, novelty=True, n_jobs=2).fit(p)
        self.ocsvm = OneClassSVM(nu=self.config.contamination, kernel="rbf", gamma="scale").fit(p)
        # Robust covariance can fail in highly singular spaces; fit only when data support it.
        try:
            self.elliptic = EllipticEnvelope(contamination=self.config.contamination, random_state=self.config.random_seed, support_fraction=0.9).fit(p)
        except Exception:
            self.elliptic = None
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.preprocess is None or self.pca is None or self.kmeans is None or self.gmm is None:
            raise RuntimeError("RegimeAnomalyFeatureLab is not fit")
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise ValueError(f"missing unsupervised input columns: {missing[:10]}")
        z = self.preprocess.transform(frame[self.columns])
        p = self.pca.transform(z)
        out = pd.DataFrame(index=frame.index)
        for j in range(p.shape[1]):
            out[f"u_pca_{j:02d}"] = p[:, j]
        out["u_kmeans_regime"] = self.kmeans.predict(p).astype(int)
        gprob = self.gmm.predict_proba(p)
        out["u_gmm_regime"] = np.argmax(gprob, axis=1).astype(int)
        out["u_gmm_confidence"] = np.max(gprob, axis=1)
        out["u_gmm_log_likelihood"] = self.gmm.score_samples(p)
        if self.isolation is not None:
            out["u_isolation_score"] = self.isolation.decision_function(p)
            out["u_isolation_anomaly"] = (self.isolation.predict(p) < 0).astype("int8")
        if self.lof is not None:
            out["u_lof_score"] = self.lof.decision_function(p)
            out["u_lof_anomaly"] = (self.lof.predict(p) < 0).astype("int8")
        if self.ocsvm is not None:
            out["u_ocsvm_score"] = self.ocsvm.decision_function(p).ravel()
            out["u_ocsvm_anomaly"] = (self.ocsvm.predict(p) < 0).astype("int8")
        if self.elliptic is not None:
            out["u_elliptic_score"] = self.elliptic.decision_function(p)
            out["u_elliptic_anomaly"] = (self.elliptic.predict(p) < 0).astype("int8")
        return out.replace([np.inf, -np.inf], np.nan)

    def save(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "unsupervised_regime_anomaly_lab.joblib", compress=3)
        meta = {
            "config": self.config.__dict__,
            "input_columns": self.columns,
            "fit_contract": "development only; no target labels",
            "interpretation": "representation/regime/anomaly features only; not direct trading authorization",
        }
        (path / "unsupervised_manifest.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def tail_enrichment(frame: pd.DataFrame, anomaly_col: str, r_col: str = "label_r_multiple", tail_r: float = -1.0) -> dict:
    """Descriptive validation/test metric: whether anomalies enrich future adverse outcomes."""
    z = frame[[anomaly_col, r_col]].dropna().copy()
    if z.empty:
        return {"rows": 0, "anomaly_rate": np.nan, "tail_rate_all": np.nan, "tail_rate_anomaly": np.nan, "enrichment": np.nan}
    anomaly = z[anomaly_col].astype(bool)
    tail = z[r_col] <= tail_r
    all_rate = float(tail.mean())
    a_rate = float(tail[anomaly].mean()) if anomaly.any() else np.nan
    return {
        "rows": int(len(z)), "anomaly_rate": float(anomaly.mean()),
        "tail_rate_all": all_rate, "tail_rate_anomaly": a_rate,
        "enrichment": float(a_rate / all_rate) if all_rate > 0 and np.isfinite(a_rate) else np.nan,
    }
