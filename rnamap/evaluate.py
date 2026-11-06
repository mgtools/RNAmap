"""Score the models on the released benchmarks, with each task's metric and protocol from the paper.

Task 1  accuracy and macro-F1 on `test` (10,016 RNAs)
Task 2  F-value (and sensitivity, PPV) per RNA, averaged, on bpRNA-new (`test_new`), TS0 and the 83 primary PDB
        chains, after nested canonical decoding at the checkpoint's threshold
Task 3  signed Spearman per assay, averaged within each RNA class and then over the three classes (the RNAGym
        leaderboard): the zero-shot coupling score over every variant; with a checkpoint, the supervised model on
        the test variants of one cross-validation fold
Task 4  Pearson R on `test` (342 variants), with a checkpoint
"""

import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

from . import predict
from .inputs import FITNESS_MIN_P
from .maps import CHANNELS, Split, codes, dotbracket_to_pairs, read_labels
from .structure import accuracy, confusion, nested_pairs
from .zeroshot import coupling_score, median_map

RNA_CLASSES = ("Ribozyme", "tRNA", "Aptamer")


def classification(root, checkpoint="ncrna_classification", device=None) -> dict:
    s = Split(root, "ncrna_classification", "test")
    table, _, _ = predict.classify(s, checkpoint, device)
    y, p = s.rows.label.to_numpy(), table.prediction.to_numpy()
    return {"n": len(s), "accuracy": float((y == p).mean()), "macro_f1": float(f1_score(y, p, average="macro"))}


def secondary_structure(root, checkpoint="secondary_structure", device=None) -> pd.DataFrame:
    rows = []
    for split, subset in (("test_new", None), ("test_ts0", "TS0"), ("test_pdb", "primary")):
        s = Split(root, "secondary_structure", split)
        probs, meta = predict.pair_probabilities(s, checkpoint, device)
        scores = []
        for k, P in enumerate(probs):
            if subset and s.rows.subset[k] != subset:
                continue
            i, j = nested_pairs(P, codes(s.sequence(k)), meta["threshold"])
            true = dotbracket_to_pairs(s.rows.label[k])
            scores.append(accuracy(*confusion(true, np.stack([i, j], 1), len(P))))
        sen, ppv, fval, mcc = np.mean(scores, 0)
        name = split if subset is None else f"{split} ({subset})"
        rows.append({"split": name, "n": len(scores), "sensitivity": sen, "ppv": ppv, "f_value": fval, "mcc": mcc})
    return pd.DataFrame(rows)


def leaderboard(labels: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Signed Spearman per assay, then the mean per RNA class and their macro mean (last row)."""
    per = []
    for a, ix in labels.groupby("assay").indices.items():
        rho = spearmanr(labels.label.to_numpy(float)[ix], score[ix]).statistic
        per.append({"assay": a, "rna_type": labels.rna_type.iloc[ix[0]], "n": len(ix),
                    "spearman": 0.0 if np.isnan(rho) else float(rho)})  # fmt: skip
    per = pd.DataFrame(per)
    cls = per.groupby("rna_type").spearman.mean().reindex(list(RNA_CLASSES))
    summary = pd.DataFrame({"rna_type": [*cls.index, "macro"], "spearman": [*cls.to_numpy(), cls.mean()]})
    return per, summary


def single_mutant_proxies(root, labels: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each assay's wild-type proxy map, the mean over channels of each channel's per-pair median over the assay's
    single mutants, read with entries below 0.01 dropped as the paper's models read them. Loads one fold at a time
    (about 2 GB each)."""
    singles = labels[labels.n_mut.astype(int) == 1]
    parts: dict[str, list] = {a: [] for a in labels.assay.unique()}
    for split in sorted(labels.split.unique()):
        s = Split(root, "variant_fitness", split)
        sub = singles[singles.split == split]
        for a, idx in sub.groupby("assay")["index"]:
            parts[a] += [s.pairs(int(k), FITNESS_MIN_P) for k in idx]
    out = {}
    for a, entries in parts.items():
        L = int(labels.length[labels.assay == a].iloc[0])
        R = np.zeros((L, L))
        for c in range(len(CHANNELS)):
            x = np.zeros((len(entries), L, L), np.float32)
            for n, (i, j, p) in enumerate(entries):
                x[n, i, j] = p[:, c]
            R += median_map(x)
        out[a] = R / len(CHANNELS)
    return out


def zeroshot_fitness(root, lam: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The coupling score s_lam of every variant against its assay's single-mutant proxy map."""
    labels = read_labels(root, "variant_fitness")
    assays = pd.read_csv(f"{root}/variant_fitness/assays.tsv", sep="\t").set_index("assay")
    proxies = single_mutant_proxies(root, labels)
    score = np.zeros(len(labels))
    for a, ix in labels.groupby("assay").indices.items():
        score[ix] = coupling_score(assays.wild_type[a], labels.sequence.iloc[ix].tolist(), proxies[a], lam)
    return leaderboard(labels, score)


def supervised_fitness(root, checkpoint, fold: int, device=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A supervised checkpoint trained on fold `fold`'s train split, scored on that fold's test variants."""
    s = Split(root, "variant_fitness", f"fold{fold}")
    table = predict.fitness(s, s.rows.assay.tolist(), checkpoint, device)
    return leaderboard(s.rows, table.score.to_numpy())


def retron(root, checkpoint, device=None) -> dict:
    s = Split(root, "retron_msdna", "test")
    table = predict.retron(s, checkpoint, device)
    r = pearsonr(table.score.to_numpy(), s.rows.label.to_numpy(float)).statistic
    return {"n": len(s), "pearson_r": float(r)}


def lam_value(text: str) -> float:
    return math.inf if text in ("inf", "infinity") else float(text)
