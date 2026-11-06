"""Task 3 zero-shot: the wild type's pair map read as the couplings of a Potts-like energy (no fitness label used).

    delta(v) = sum_{i<j} R_ij [c(wt_i, wt_j) - c(v_i, v_j)]
    s_lam(v) = -n_mut(v) - lam * delta(v)            (higher = predicted fitter)

R is the wild type's pair map (the mean of the five channels), c(x, y) = 1 for the six canonical pairs G-C, C-G, A-U,
U-A, G-U, U-G and 0 otherwise. delta is the expected number of wild-type pairs a variant breaks, net of restorations:
a G-C -> A-U double mutant costs nothing. The paper's score is s_1 (a broken pair costs as much as a substitution);
s_inf keeps the map term alone.

The wild-type map: for your own RNA, run `rnamap featurize` on the wild type and use its map directly. On RNAGym the
paper had no wild-type maps and used a proxy, the per-pair median of each channel over the assay's single mutants
(`median_map`); the exact ViennaRNA wild type changes the RNAfold channel's score by +0.007.
"""

import math

import numpy as np

from .maps import BASES, codes

CANONICAL = ("GC", "CG", "AU", "UA", "GU", "UG")


def parse_mutant(m: str) -> list[tuple[str, int, str]]:
    """'C10A,U56C' -> [('C', 9, 'A'), ('U', 55, 'C')]: wild-type base, 0-based position, new base (T read as U)."""
    out = []
    for s in m.replace(" ", "").upper().replace("T", "U").split(","):
        if s:
            out.append((s[0], int(s[1:-1]) - 1, s[-1]))
    return out


def apply_mutant(wt: str, m: str) -> str:
    """The variant sequence of mutant string `m` on wild type `wt`; checks every named wild-type base."""
    s = list(wt.upper().replace("T", "U"))
    for a, k, b in parse_mutant(m):
        if s[k] != a:
            raise ValueError(f"{m}: position {k + 1} of the wild type is {s[k]}, not {a}")
        if b not in BASES:
            raise ValueError(f"{m}: {b} is not one of {BASES}")
        s[k] = b
    return "".join(s)


def pair_table() -> np.ndarray:
    """(5, 5) c(x, y) on base codes 0-4 (0 = other letter, 1-4 = A, C, G, U)."""
    c = np.zeros((5, 5), np.float64)
    for x, y in CANONICAL:
        c[BASES.index(x) + 1, BASES.index(y) + 1] = 1
    return c


def pairs_broken(R: np.ndarray, wt: str, variants: list[str]) -> np.ndarray:
    """delta for each variant: R (L, L) the wild-type map, `wt` and `variants` sequences of length L."""
    c, w = pair_table(), codes(wt.upper().replace("T", "U"))
    L = len(w)
    iu, ju = np.triu_indices(L, 1)
    r = np.asarray(R, np.float64)[iu, ju]
    keep = r > 0
    iu, ju, r = iu[keep], ju[keep], r[keep]
    base = c[w[iu], w[ju]]
    out = np.zeros(len(variants))
    for n, v in enumerate(variants):
        x = codes(v.upper().replace("T", "U"))
        assert len(x) == L, f"variant {n} has length {len(x)}, the wild type {L}"
        out[n] = (r * (base - c[x[iu], x[ju]])).sum()
    return out


def n_mutations(wt: str, variants: list[str]) -> np.ndarray:
    w = np.frombuffer(wt.upper().replace("T", "U").encode(), np.uint8)
    return np.array([(np.frombuffer(v.upper().replace("T", "U").encode(), np.uint8) != w).sum() for v in variants])


def coupling_score(wt: str, variants: list[str], R: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """s_lam of each variant; `lam` = inf gives -delta."""
    delta = pairs_broken(R, wt, variants)
    return -delta if math.isinf(lam) else -n_mutations(wt, variants) - lam * delta


def median_map(maps: np.ndarray) -> np.ndarray:
    """Per-pair median over the first axis of (n, ...) maps (the mean of the two middle values when n is even)."""
    s = np.sort(maps, axis=0)
    n = len(s)
    return (s[(n - 1) // 2] + s[n // 2]) / 2
