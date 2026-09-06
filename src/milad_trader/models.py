"""Train-only normalization and compact reproducible supervised baselines."""
import copy
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def windows(x, indices, lookback):
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0 or indices.min() < lookback-1 or indices.max() >= len(x):
        raise ValueError("Invalid sequence boundaries")
    return np.stack([x[j-lookback+1:j+1] for j in indices]).astype(np.float32)


def build_network(kind, n_features, lookback):
    import torch
    from torch import nn

    if kind == "lstm":
        # Modules are defined at top level in torch_models for checkpoint portability.
        from .torch_models import LSTMClassifier
        return LSTMClassifier(n_features)
    if kind == "cnn":
        from .torch_models import CNNClassifier
        return CNNClassifier(n_features)
    if kind == "transformer":
        from .torch_models import PatchClassifier
        return PatchClassifier(n_features, lookback)
    raise ValueError(kind)


class Predictor:
    def __init__(self, kind, lookback=24, seed=42, epochs=20):
        self.kind, self.lookback, self.seed, self.epochs = kind, lookback, seed, epochs
        self.scaler = StandardScaler()
        self.constant = None
        self.validation_losses = []

    def fit(self, values, labels, train_indices, val_indices):
        seed_everything(self.seed)
        if not np.isfinite(labels[train_indices]).all() or not np.isfinite(labels[val_indices]).all():
            raise ValueError("Unknown future labels cannot be used for fitting")
        train_end = int(max(train_indices)) + 1
        # Window context is also in training history; nothing after train_end is fitted.
        self.scaler.fit(values[:train_end])
        x = self.scaler.transform(values)
        train_x = windows(x, train_indices, self.lookback)
        val_x = windows(x, val_indices, self.lookback)
        y, vy = labels[train_indices].astype(int), labels[val_indices].astype(int)
        if len(np.unique(y)) == 1:
            self.constant = float(y[0])
            return self
        if self.kind == "random_forest":
            self.model = RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_leaf=10,
                                                class_weight="balanced", random_state=self.seed, n_jobs=1)
            self.model.fit(train_x.reshape(len(y),-1), y)
        elif self.kind == "xgboost":
            from xgboost import XGBClassifier
            self.model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=.04,
                                       subsample=.8, colsample_bytree=.8, reg_lambda=5,
                                       eval_metric="logloss", early_stopping_rounds=15,
                                       random_state=self.seed, n_jobs=1)
            self.model.fit(train_x.reshape(len(y),-1), y,
                           eval_set=[(val_x.reshape(len(vy),-1),vy)], verbose=False)
        else:
            import torch
            from torch import nn
            self.model = build_network(self.kind, values.shape[1], self.lookback)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=.001, weight_decay=.01)
            criterion = nn.BCEWithLogitsLoss()
            tx, ty = torch.from_numpy(train_x), torch.tensor(y,dtype=torch.float32)
            vx, vtarget = torch.from_numpy(val_x), torch.tensor(vy,dtype=torch.float32)
            best, patience, state = float("inf"), 0, None
            generator = torch.Generator().manual_seed(self.seed)
            for _ in range(self.epochs):
                self.model.train()
                for idx in torch.randperm(len(tx), generator=generator).split(64):
                    optimizer.zero_grad()
                    loss = criterion(self.model(tx[idx]), ty[idx])
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(),1.)
                    optimizer.step()
                self.model.eval()
                with torch.no_grad():
                    value = float(criterion(self.model(vx),vtarget))
                self.validation_losses.append(value)
                if value < best-1e-6:
                    best, patience, state = value, 0, copy.deepcopy(self.model.state_dict())
                else:
                    patience += 1
                    if patience >= 5:
                        break
            self.model.load_state_dict(state)
            self.model.eval()
        return self

    def predict(self, values, indices):
        if self.constant is not None:
            return np.full(len(indices),self.constant)
        x = windows(self.scaler.transform(values), indices, self.lookback)
        if self.kind in ("random_forest", "xgboost"):
            return self.model.predict_proba(x.reshape(len(x),-1))[:,1]
        import torch
        self.model.eval()
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(self.model(torch.from_numpy(batch))).numpy()
                                   for batch in np.array_split(x,max(1,int(np.ceil(len(x)/256))))])
