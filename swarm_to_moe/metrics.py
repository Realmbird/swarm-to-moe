"""Routing and statistics metrics. numpy in / numpy out; routing indices may be torch tensors."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.stats import binomtest

from .partition import Partition


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _mask_np(mask, B, T) -> np.ndarray:
    if mask is None:
        return np.ones((B, T), dtype=bool)
    m = _np(mask).astype(bool)
    if m.shape != (B, T):
        raise ValueError(f"mask shape {m.shape} != {(B, T)}")
    return m


# ---------------------------------------------------------------------------- usage / ownership


def usage_counts(indices, num_experts: int, mask=None) -> np.ndarray:
    """Count top-k selections per expert. indices [L,B,T,k] -> counts [B,L,E] over (masked) tokens."""
    idx = _np(indices).astype(np.int64)
    L, B, T, k = idx.shape
    m = _mask_np(mask, B, T)
    onehot_counts = np.zeros((B, L, num_experts), dtype=np.int64)
    w = np.broadcast_to(m[None, :, :, None], idx.shape)
    for l in range(L):
        for b in range(B):
            sel = idx[l, b][w[l, b]]
            onehot_counts[b, l] = np.bincount(sel.ravel(), minlength=num_experts)[:num_experts]
    return onehot_counts


def owner_fractions(indices, partition: Partition, agents: Sequence[str | None], mask=None) -> dict:
    """Fractions of top-k slots (over masked tokens) on own-private ('match'), other roles' private ('other')
    and shared experts. Returns dict of [B, L] arrays. Rows with agent None get NaN for match/other."""
    idx = _np(indices).astype(np.int64)
    L, B, T, k = idx.shape
    if len(agents) != B:
        raise ValueError("len(agents) must equal batch size")
    counts = usage_counts(idx, partition.num_experts, mask).astype(np.float64)  # [B,L,E]
    om = partition.owner_matrix()  # [L,E]
    total = counts.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        shared = (counts * (om == 0)[None]).sum(-1) / total
        match = np.full((B, L), np.nan)
        other = np.full((B, L), np.nan)
        for b, a in enumerate(agents):
            if a is None:
                continue
            code = partition.roles.index(a) + 1
            match[b] = (counts[b] * (om == code)).sum(-1) / total[b]
            other[b] = (counts[b] * ((om > 0) & (om != code))).sum(-1) / total[b]
    return {"match": match, "other": other, "shared": shared}


def role_preference(indices, partition: Partition, agents: Sequence[str | None], mask=None) -> np.ndarray:
    """match - other/(R-1), [B, L]. 0 under role-blind routing over equal-sized private sets."""
    f = owner_fractions(indices, partition, agents, mask)
    R = len(partition.roles)
    return f["match"] - f["other"] / (R - 1)


# ---------------------------------------------------------------------------- distances


def routing_jsd(p, q) -> np.ndarray:
    """Per-layer Jensen-Shannon divergence (natural log, <= ln 2) between [L,E] usage arrays (rows normalised)."""
    p = _np(p).astype(np.float64)
    q = _np(q).astype(np.float64)
    p = p / p.sum(-1, keepdims=True)
    q = q / q.sum(-1, keepdims=True)
    m = 0.5 * (p + q)

    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(a > 0, a * np.log(a / b), 0.0)
        return t.sum(-1)

    return np.clip(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0, np.log(2))


def routing_jaccard(a, b, mask=None) -> float:
    """Pooled Jaccard of top-k sets: total |A & B| / total |A | B| over all (masked) layer/row/token triples."""
    a = _np(a).astype(np.int64)
    b = _np(b).astype(np.int64)
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    L, B, T, k = a.shape
    m = np.broadcast_to(_mask_np(mask, B, T)[None], (L, B, T))
    E = int(max(a.max(), b.max())) + 1
    a2, b2 = a[m], b[m]  # [N, k]
    n = a2.shape[0]
    if n == 0:
        return float("nan")
    ha = np.zeros((n, E), dtype=bool)
    hb = np.zeros((n, E), dtype=bool)
    rows = np.arange(n)[:, None]
    ha[rows, a2] = True
    hb[rows, b2] = True
    return float((ha & hb).sum() / (ha | hb).sum())


# ---------------------------------------------------------------------------- statistics


def paired_bootstrap_ci(a, b, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Mean of (a - b) with a percentile bootstrap CI resampling pairs."""
    a = _np(a).astype(np.float64)
    b = _np(b).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError("a and b must be paired")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def mcnemar_holm(correct: Mapping[str, np.ndarray], reference: str, alpha: float = 0.05) -> dict:
    """Exact (binomial) two-sided McNemar of each arm vs ``reference``, Holm step-down over all arms.

    b = reference right & arm wrong, c = reference wrong & arm right."""
    ref = _np(correct[reference]).astype(bool)
    names = [n for n in correct if n != reference]
    res = {}
    for n in names:
        x = _np(correct[n]).astype(bool)
        if x.shape != ref.shape:
            raise ValueError(f"{n}: not paired with reference")
        b = int((ref & ~x).sum())
        c = int((~ref & x).sum())
        p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
        res[n] = {"p_raw": float(p), "b": b, "c": c, "n_discordant": b + c}
    m = len(names)
    order = sorted(names, key=lambda n: res[n]["p_raw"])
    running = 0.0
    still = True
    for i, n in enumerate(order):
        adj = min(1.0, (m - i) * res[n]["p_raw"])
        running = max(running, adj)
        res[n]["p_holm"] = float(running)
        still = still and running < alpha
        res[n]["significant"] = bool(still)
    return res


def cv_splits(y, folds: int = 5, seed: int = 0, groups=None) -> list:
    """Fold list shared by the routing and text classifiers. Without groups: StratifiedKFold (shuffled,
    seeded). With groups (e.g. input index): StratifiedGroupKFold (a GroupKFold that also balances
    classes), so no group appears in both train and test."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    y = np.asarray(y)
    dummy = np.zeros((len(y), 1))
    if groups is None:
        return list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(dummy, y))
    return list(StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(dummy, y, np.asarray(groups)))


def role_id_cv(X, y, folds: int = 5, seed: int = 0, groups=None, return_predictions: bool = False):
    """k-fold logistic regression (standardised features). Returns (accuracy, chance=1/n_classes).
    Folds from ``cv_splits`` (grouped when ``groups`` is given)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = _np(X).astype(np.float64)
    y = np.asarray(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pred = cross_val_predict(clf, X, y, cv=cv_splits(y, folds, seed, groups))
    acc = float((pred == y).mean())
    chance = 1.0 / len(np.unique(y))
    if return_predictions:
        return acc, chance, pred
    return acc, chance


def text_id_cv(texts: Sequence[str], y, folds: int = 5, seed: int = 0, groups=None):
    """Text baseline with the same folds as ``role_id_cv``: TF-IDF (word 1-2 grams) + logistic regression."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline

    y = np.asarray(y)
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                        LogisticRegression(max_iter=2000))
    pred = cross_val_predict(clf, list(texts), y, cv=cv_splits(y, folds, seed, groups))
    return float((pred == y).mean()), 1.0 / len(np.unique(y))


def routing_jaccard_counts(a, b, mask=None) -> tuple[int, int]:
    """(total intersection, total union) of top-k sets, for pooling ``routing_jaccard`` across batches."""
    a = _np(a).astype(np.int64)
    b = _np(b).astype(np.int64)
    L, B, T, k = a.shape
    m = np.broadcast_to(_mask_np(mask, B, T)[None], (L, B, T))
    a2, b2 = a[m], b[m]
    if a2.shape[0] == 0:
        return 0, 0
    inter = int((a2[:, :, None] == b2[:, None, :]).any(-1).sum())
    return inter, int(2 * k * a2.shape[0] - inter)


def l1_per_layer(counts) -> np.ndarray:
    """Normalise [..., L, E] counts to per-layer distributions (rows with zero mass stay zero)."""
    c = _np(counts).astype(np.float64)
    s = c.sum(-1, keepdims=True)
    return np.divide(c, s, out=np.zeros_like(c), where=s > 0)


def mean_pairwise_kappa(correct: Mapping[str, np.ndarray]) -> float:
    """Mean Cohen's kappa of error agreement over all pairs of arms."""
    from itertools import combinations

    from sklearn.metrics import cohen_kappa_score

    names = list(correct)
    ks = []
    for a, b in combinations(names, 2):
        xa, xb = _np(correct[a]).astype(int), _np(correct[b]).astype(int)
        if len(np.unique(np.concatenate([xa, xb]))) < 2:
            continue
        ks.append(cohen_kappa_score(xa, xb))
    return float(np.mean(ks)) if ks else float("nan")
