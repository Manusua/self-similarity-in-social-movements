"""Adjusted mutual information without scikit-learn dependency."""

from __future__ import annotations

import numpy as np


def _entropy_from_labels(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _contingency_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    classes_true = np.unique(y_true)
    classes_pred = np.unique(y_pred)
    lookup_true = {c: i for i, c in enumerate(classes_true)}
    lookup_pred = {c: i for i, c in enumerate(classes_pred)}
    cont = np.zeros((len(classes_true), len(classes_pred)), dtype=float)
    for a, b in zip(y_true, y_pred):
        cont[lookup_true[a], lookup_pred[b]] += 1
    return cont


def _expected_mutual_information(contingency: np.ndarray) -> float:
    n = contingency.sum()
    if n == 0:
        return 0.0
    row_sum = contingency.sum(axis=1)
    col_sum = contingency.sum(axis=0)
    emi = 0.0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            n_ij = contingency[i, j]
            if n_ij == 0:
                continue
            emi += (n_ij / n) * np.log((n * n_ij) / (row_sum[i] * col_sum[j] + 1e-12))
    return emi


def adjusted_mutual_info_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cont = _contingency_matrix(y_true, y_pred)
    n = cont.sum()
    if n == 0:
        return 1.0
    outer = np.outer(cont.sum(axis=1), cont.sum(axis=0))
    mi = 0.0
    for i in range(cont.shape[0]):
        for j in range(cont.shape[1]):
            n_ij = cont[i, j]
            if n_ij == 0:
                continue
            mi += (n_ij / n) * np.log((n * n_ij) / (outer[i, j] + 1e-12))
    emi = _expected_mutual_information(cont)
    h_true = _entropy_from_labels(y_true)
    h_pred = _entropy_from_labels(y_pred)
    normalizer = 0.5 * (h_true + h_pred)
    denom = normalizer - emi
    if abs(denom) < 1e-12:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0
    return float((mi - emi) / denom)
