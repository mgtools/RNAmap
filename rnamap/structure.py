"""Decode pair probabilities into one secondary structure, and score structures as the paper does.

`nested_pairs` keeps, among the canonical pairs (AU, GC, GU) with j - i >= 4 and p > t, the pseudoknot-free set that
maximises the sum of p - t (a Nussinov-style dynamic programme). The Task 2 checkpoint's t (0.2) was tuned on the
held-out validation families. `accuracy` gives sensitivity, PPV, F-value and MCC by exact base-pair matching.
"""

import math

import numba
import numpy as np

from .models import MIN_SEP

CANONICAL = ((1, 4), (4, 1), (2, 3), (3, 2), (3, 4), (4, 3))  # AU UA CG GC GU UG in base codes A1 C2 G3 U4


def canonical_mask(codes: np.ndarray) -> np.ndarray:
    """(L, L) bool: which pairs of the sequence (base codes, 0 for other letters) are AU, GC or GU."""
    table = np.zeros((5, 5), bool)
    for a, b in CANONICAL:
        table[a, b] = True
    return table[codes[:, None], codes[None, :]]


@numba.njit(cache=True)
def _nested_dp(L: int, ptr: np.ndarray, partner: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Maximum-weight nested pairing. Candidate partners k > i of nucleotide i are partner[ptr[i]:ptr[i + 1]] with
    positive weights; returns the chosen partner of each nucleotide that opens a pair, -1 elsewhere."""
    best = np.zeros((L + 1, L + 1))  # best[i, j]: best weight on nucleotides i .. j - 1
    choice = np.full((L + 1, L + 1), -1, np.int32)
    for i in range(L - 1, -1, -1):
        for j in range(i + 1, L + 1):
            b, c = best[i + 1, j], -1  # i unpaired
            for e in range(ptr[i], ptr[i + 1]):
                k = partner[e]
                if k < j:
                    v = weight[e] + best[i + 1, k] + best[k + 1, j]
                    if v > b:
                        b, c = v, k
            best[i, j], choice[i, j] = b, c
    out = np.full(L, -1, np.int32)
    stack = [(0, L)]
    while stack:
        i, j = stack.pop()
        while i < j:
            k = choice[i, j]
            if k < 0:
                i += 1
            else:
                out[i] = k
                stack.append((i + 1, k))
                i = k + 1
    return out


def nested_pairs(P: np.ndarray, codes: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """(i, j) of the nested canonical structure that maximises the sum of p - t over pairs with p > t."""
    i, j = np.triu_indices(len(P), k=MIN_SEP)
    p = np.asarray(P, np.float64)[i, j]
    keep = (p > t) & canonical_mask(codes)[i, j]
    i, j, w = i[keep], j[keep], p[keep] - t  # triu_indices sorts by i, as the CSR layout needs
    ptr = np.searchsorted(i, np.arange(len(P) + 1)).astype(np.int64)
    opener = _nested_dp(len(P), ptr, j.astype(np.int64), w)
    a = np.flatnonzero(opener >= 0)
    return a, opener[a].astype(np.int64)


def confusion(true: np.ndarray, pred: np.ndarray, L: int) -> tuple[int, int, int, int]:
    """(tp, tn, fp, fn) of predicted pairs against true pairs, both (n, 2) arrays of 0-based i < j.

    A pair counts by exact match. As in MXfold2's scorer, a reference nucleotide paired to a different partner than
    predicted is one false positive and one false negative.
    """
    ref = np.zeros(L + 1, np.int64)
    prd = np.zeros(L + 1, np.int64)
    for arr, (a, b) in ((ref, true.T), (prd, pred.T)):
        arr[a + 1], arr[b + 1] = b + 1, a + 1
    tp = fp = fn = 0
    for i in range(1, L + 1):
        j1, j2 = ref[i], prd[i]
        if j1 > 0 and i < j1:
            if j1 == j2:
                tp += 1
            elif j2 > 0 and i < j2:
                fp += 1
                fn += 1
            else:
                fn += 1
        elif j2 > 0 and i < j2:
            fp += 1
    return tp, L * (L - 1) // 2 - tp - fp - fn, fp, fn


def accuracy(tp: int, tn: int, fp: int, fn: int) -> tuple[float, float, float, float]:
    """Sensitivity, PPV, F-value and MCC (0 where undefined; an RNA without true pairs scores 0)."""
    sen = tp / (tp + fn) if tp + fn > 0 else 0.0
    ppv = tp / (tp + fp) if tp + fp > 0 else 0.0
    fval = 2 * sen * ppv / (sen + ppv) if sen + ppv > 0 else 0.0
    d = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn) - (fp * fn)) / math.sqrt(d) if d > 0 else 0.0
    return sen, ppv, fval, mcc
