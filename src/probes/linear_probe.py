"""
Week 1-2: Linear Probing Classifier

Used across Exp 1 (layer-wise probing) and Exp 5 (token-position grid).
A simple logistic regression probe trained on frozen BERT representations
to test whether a grammatical feature is linearly decodable at a given layer.

Following Hewitt & Liang (2019), we also implement a control task probe
to ensure probing accuracy exceeds what a probe memorizing word identity
would achieve.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from dataclasses import dataclass


@dataclass
class ProbeResult:
    """Results from a single probing experiment."""
    layer: int
    accuracy: float
    f1: float
    accuracy_std: float
    f1_std: float
    control_accuracy: float = 0.0  # from control task (selectivity)
    selectivity: float = 0.0       # accuracy - control_accuracy
    n_samples: int = 0


class LinearProbe:
    """
    Logistic regression probe for binary grammatical classification.

    Given hidden states from a specific BERT layer, trains a probe to
    distinguish grammatical vs. ungrammatical sentences.
    """

    def __init__(self, max_iter: int = 1000, C: float = 1.0, n_folds: int = 5):
        self.max_iter = max_iter
        self.C = C
        self.n_folds = n_folds

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        layer: int,
    ) -> ProbeResult:
        """
        Train probe with k-fold cross-validation.

        Args:
            X: Feature matrix, shape (n_samples, hidden_dim).
               Typically the [CLS] token representation or mean-pooled.
            y: Binary labels, shape (n_samples,). 1=grammatical, 0=ungrammatical.
            layer: Which BERT layer these representations come from (for logging).

        Returns:
            ProbeResult with accuracy and F1 across folds.
        """
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        accs, f1s = [], []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            clf = LogisticRegression(
                max_iter=self.max_iter,
                C=self.C,
                solver="lbfgs",
                random_state=42,
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)

            accs.append(accuracy_score(y_val, y_pred))
            n_classes = len(set(y))
            avg = "binary" if n_classes == 2 else "weighted"
            f1s.append(f1_score(y_val, y_pred, average=avg))

        return ProbeResult(
            layer=layer,
            accuracy=np.mean(accs),
            f1=np.mean(f1s),
            accuracy_std=np.std(accs),
            f1_std=np.std(f1s),
            n_samples=len(y),
        )

    def control_task(
        self,
        X: np.ndarray,
        y_random: np.ndarray,
        layer: int,
    ) -> float:
        """
        Control task probe (Hewitt & Liang 2019).

        Trains the same probe on randomly assigned labels.
        If the probe achieves high accuracy on the control task,
        the representation is too expressive and selectivity is low.

        Returns:
            Control accuracy (averaged across folds).
        """
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        accs = []
        for train_idx, val_idx in skf.split(X, y_random):
            clf = LogisticRegression(
                max_iter=self.max_iter, C=self.C, solver="lbfgs", random_state=42,
            )
            clf.fit(X[train_idx], y_random[train_idx])
            y_pred = clf.predict(X[val_idx])
            accs.append(accuracy_score(y_random[val_idx], y_pred))
        return np.mean(accs)


def prepare_probing_data(
    hidden_states_dir,
    metadata,
    layer: int,
    pooling: str = "cls",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load hidden states for a given layer and prepare (X, y) for probing.

    For each minimal pair, we get two data points:
      - good sentence → label 1
      - bad sentence  → label 0

    Args:
        hidden_states_dir: Path to directory with pair_XXXX.npz files.
        metadata: List of metadata dicts (from metadata.json).
        layer: BERT layer index (0=embeddings, 1-12=transformer layers).
        pooling: "cls" for [CLS] token, "mean" for mean pooling.

    Returns:
        X: (2*n_pairs, hidden_dim)
        y: (2*n_pairs,)
    """
    X_list, y_list = [], []

    for entry in metadata:
        npz = np.load(hidden_states_dir / f"pair_{entry['idx']:04d}.npz")

        good_hidden = npz["good_hidden"][layer]  # (seq_len, 768)
        bad_hidden = npz["bad_hidden"][layer]

        if pooling == "cls":
            good_vec = good_hidden[0]   # [CLS] token
            bad_vec = bad_hidden[0]
        elif pooling == "mean":
            good_vec = good_hidden.mean(axis=0)
            bad_vec = bad_hidden.mean(axis=0)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        X_list.append(good_vec)
        y_list.append(1)
        X_list.append(bad_vec)
        y_list.append(0)

    return np.array(X_list), np.array(y_list)
