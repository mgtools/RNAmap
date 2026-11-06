"""Five-channel base-pair maps in the benchmark format, for the released benchmarks and for your own RNAs alike.

A set of RNAs is two files: a `.npz` of maps and a table of the RNAs (`id`, `sequence`, ...). The `.npz` holds plain
arrays (no pickles):

    id        (n,) str        RNA id, as in the table
    length    (n,) int32      sequence length L
    channels  (5,) str        RNAfold, CONTRAfold, EternaFold, RibonanzaNet-SS, ERNIE-RNA
    offsets   (n + 1,) int64  the entries of RNA k are offsets[k] : offsets[k + 1]
    i, j      (nnz,) uint16   0-based positions, i < j
    p         (nnz, 5) f16    the probability that i and j pair, one column per channel

The maps are symmetric, so only i < j is stored. An RNA's entries are the union of the pairs any predictor stored, and
a channel is 0 where its own value was not stored (below 1e-3; ERNIE-RNA below 2e-3).

    from rnamap.maps import Split, Maps
    s = Split("data", "secondary_structure", "test_new")   # a released benchmark split
    m = Maps.load("my_rnas")                                # my_rnas.npz + my_rnas.tsv from `rnamap featurize`
    x = m.image(0)                                          # (5, L, L) float32
"""

from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ("RNAfold", "CONTRAfold", "EternaFold", "RibonanzaNet-SS", "ERNIE-RNA")
BASES = "ACGU"
OPEN, CLOSE = "([{<" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ", ")]}>" + "abcdefghijklmnopqrstuvwxyz"


class Maps:
    """The maps of n RNAs (`z`, the arrays of the .npz) and their table (`rows`, one row per RNA, in file order)."""

    def __init__(self, z: dict, rows: pd.DataFrame):
        self.z, self.rows = z, rows.reset_index(drop=True)
        assert (np.asarray(z["id"]) == self.rows.id.to_numpy()).all(), "the .npz and the table disagree"
        assert tuple(z["channels"]) == CHANNELS, f"unexpected channels {tuple(z['channels'])}"

    @classmethod
    def load(cls, path) -> "Maps":
        """`path`.npz with `path`.tsv (written by `rnamap featurize`); `path` may end in .npz."""
        path = Path(path)
        stem = path.with_suffix("") if path.suffix == ".npz" else path
        z = _read_npz(stem.with_suffix(".npz"))
        rows = pd.read_csv(stem.with_suffix(".tsv"), sep="\t", dtype={"id": str}, keep_default_na=False)
        return cls(z, rows)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray(self.z["length"], np.int64)

    def sequence(self, k: int) -> str:
        return self.rows.sequence[k].upper().replace("T", "U")

    def pairs(self, k: int, min_p: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stored entries of RNA k: i, j (int64, i < j) and p (n, 5) float32. Values below `min_p` are set to 0."""
        a, b = self.z["offsets"][k], self.z["offsets"][k + 1]
        p = self.z["p"][a:b].astype(np.float32)
        if min_p > 0:
            p = np.where(p >= min_p, p, 0)
        return self.z["i"][a:b].astype(np.int64), self.z["j"][a:b].astype(np.int64), p

    def image(self, k: int, min_p: float = 0.0) -> np.ndarray:
        """(5, L, L) float32 map of RNA k, symmetric, 0 off the stored pairs."""
        L = int(self.lengths[k])
        i, j, p = self.pairs(k, min_p)
        x = np.zeros((len(CHANNELS), L, L), np.float32)
        x[:, i, j] = p.T
        x[:, j, i] = p.T
        return x

    def save(self, path) -> None:
        """Write `path`.npz and `path`.tsv."""
        stem = Path(path).with_suffix("") if Path(path).suffix == ".npz" else Path(path)
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(stem.with_suffix(".npz"), **self.z)
        self.rows.to_csv(stem.with_suffix(".tsv"), sep="\t", index=False)


class Split(Maps):
    """One split of a released benchmark: `<root>/<task>/<split>.npz` and its rows of `<root>/<task>/labels.tsv`."""

    def __init__(self, root, task: str, split: str):
        labels = read_labels(root, task)
        rows = labels[labels.split == split].sort_values("index")
        super().__init__(_read_npz(Path(root) / task / f"{split}.npz"), rows)


def read_labels(root, task: str) -> pd.DataFrame:
    """labels.tsv of a task, every column as stored (strings; empty cells are '')."""
    t = pd.read_csv(Path(root) / task / "labels.tsv", sep="\t", dtype=str, keep_default_na=False)
    for c in ("index", "length"):
        t[c] = t[c].astype(int)
    return t


def from_dense(ids: list[str], seqs: list[str], maps: list[np.ndarray], floors=(1e-3,) * 4 + (2e-3,)) -> Maps:
    """Pack dense (5, L, L) maps into the benchmark format: pairs i < j where any channel reaches its floor, and each
    channel 0 where it is below its own floor."""
    ii, jj, pp, offsets = [], [], [], [0]
    floor = np.asarray(floors, np.float32)[:, None]
    for x in maps:
        i, j = np.triu_indices(x.shape[-1], 1)
        p = x[:, i, j]
        p = np.where(p >= floor, p, 0)
        keep = (p > 0).any(0)
        ii.append(i[keep].astype(np.uint16))
        jj.append(j[keep].astype(np.uint16))
        pp.append(p[:, keep].T.astype(np.float16))
        offsets.append(offsets[-1] + int(keep.sum()))
    z = {
        "id": np.array(ids),
        "length": np.array([len(s) for s in seqs], np.int32),
        "channels": np.array(CHANNELS),
        "offsets": np.array(offsets, np.int64),
        "i": np.concatenate(ii) if ii else np.zeros(0, np.uint16),
        "j": np.concatenate(jj) if jj else np.zeros(0, np.uint16),
        "p": np.concatenate(pp) if pp else np.zeros((0, len(CHANNELS)), np.float16),
    }
    return Maps(z, pd.DataFrame({"id": ids, "length": z["length"], "sequence": seqs}))


def _read_npz(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def codes(seq: str) -> np.ndarray:
    """(L,) uint8 base codes: 1-4 for A, C, G, U (T read as U, any case), 0 for any other letter."""
    lut = np.zeros(256, np.uint8)
    for code, letters in enumerate(("Aa", "Cc", "Gg", "UuTt"), start=1):
        lut[np.frombuffer(letters.encode(), np.uint8)] = code
    return lut[np.frombuffer(seq.encode(), np.uint8)]


def dotbracket_to_pairs(db: str) -> np.ndarray:
    """(n, 2) 0-based pairs i < j of a bpRNA-style dot-bracket string: () [] {} <> then Aa Bb ... for pseudoknots."""
    match, stacks, pairs = dict(zip(CLOSE, OPEN, strict=True)), {c: [] for c in OPEN}, []
    for k, c in enumerate(db):
        if c in stacks:
            stacks[c].append(k)
        elif c in match:
            if not stacks[match[c]]:
                raise ValueError(f"unbalanced '{c}' at position {k}")
            pairs.append((stacks[match[c]].pop(), k))
    if any(stacks.values()):
        raise ValueError("unbalanced brackets")
    return np.array(sorted(pairs), np.int64).reshape(-1, 2)


def pairs_to_dotbracket(i: np.ndarray, j: np.ndarray, L: int) -> str:
    """Dot-bracket string of pairs (i < j); a nested structure uses only '(' and ')'."""
    s = ["."] * L
    order = np.argsort(i)
    placed: list[tuple[int, int, int]] = []  # (i, j, bracket level)
    for a, b in zip(np.asarray(i)[order].tolist(), np.asarray(j)[order].tolist(), strict=True):
        level = 0
        while any(lv == level and (x < a < y < b or a < x < b < y) for x, y, lv in placed):
            level += 1
        placed.append((a, b, level))
        s[a], s[b] = OPEN[level], CLOSE[level]
    return "".join(s)
