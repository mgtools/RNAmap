"""Inference for the four tasks on a `Maps` set (a released split or your own featurized RNAs)."""

import contextlib

import numpy as np
import pandas as pd
import torch

from . import checkpoints
from .inputs import classification_input, fitness_input, retron_input
from .maps import Maps, codes, pairs_to_dotbracket
from .models import MIN_SEP, with_sequence
from .structure import nested_pairs


def _autocast(device: str):
    """bf16 on a GPU, as the paper's models were evaluated; plain fp32 on the CPU."""
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _batches(n: int, size: int):
    return (np.arange(a, min(a + size, n)) for a in range(0, n, size))


@torch.no_grad()
def classify(m: Maps, checkpoint="ncrna_classification", device: str | None = None, batch_size: int = 16):
    """Task 1: (table of id, predicted class, its probability; (n, 16) probabilities; the class names)."""
    device = device or checkpoints.default_device()
    net, meta = checkpoints.load(checkpoint, device)
    classes = list(meta["classes"])
    probs = []
    for ix in _batches(len(m), batch_size):
        xs, cs = zip(*(classification_input(m, k) for k in ix), strict=True)
        x = torch.from_numpy(np.stack(xs)).to(device)
        x = with_sequence(x, torch.from_numpy(np.stack(cs)).to(device)).contiguous(memory_format=torch.channels_last)
        with _autocast(device):
            probs.append(torch.softmax(net(x).float(), 1).cpu().numpy())
    P = np.concatenate(probs) if probs else np.zeros((0, len(classes)), np.float32)
    best = P.argmax(1)
    table = pd.DataFrame({"id": m.rows.id, "prediction": [classes[b] for b in best], "probability": P.max(1)})
    return table, P, classes


@torch.no_grad()
def pair_probabilities(m: Maps, checkpoint="secondary_structure", device: str | None = None):
    """Task 2: each RNA's (L, L) pair probabilities (pairs with j - i < 4 set to 0), and the checkpoint metadata."""
    device = device or checkpoints.default_device()
    net, meta = checkpoints.load(checkpoint, device)
    out = []
    for k in range(len(m)):
        x = torch.from_numpy(m.image(k))[None].to(device)
        L = x.shape[-1]
        with _autocast(device):
            P = torch.sigmoid(net(x, torch.tensor([L], device=device)).float())[0].cpu().numpy()
        P[np.abs(np.subtract.outer(np.arange(L), np.arange(L))) < MIN_SEP] = 0
        out.append(P)
    return out, meta


def fold(m: Maps, checkpoint="secondary_structure", device: str | None = None, threshold: float | None = None):
    """Task 2: (table of id, sequence, dot-bracket structure; the pair-probability matrices)."""
    probs, meta = pair_probabilities(m, checkpoint, device)
    t = meta["threshold"] if threshold is None else threshold
    structures = []
    for k, P in enumerate(probs):
        i, j = nested_pairs(P, codes(m.sequence(k)), t)
        structures.append(pairs_to_dotbracket(i, j, len(P)))
    return pd.DataFrame({"id": m.rows.id, "sequence": [m.sequence(k) for k in range(len(m))],
                         "structure": structures}), probs  # fmt: skip


@torch.no_grad()
def fitness(m: Maps, assay: str | list[str], checkpoint, device: str | None = None, batch_size: int = 64):
    """Task 3 (supervised, M2): a score per variant of one of the 31 RNAGym assays (higher = fitter; comparable within
    an assay only). `assay` names the assay of every variant, or of each one."""
    device = device or checkpoints.default_device()
    net, meta = checkpoints.load(checkpoint, device)
    assays, lmax = list(meta["assays"]), net.add_w.shape[1]
    names = [assay] * len(m) if isinstance(assay, str) else list(assay)
    unknown = sorted(set(names) - set(assays))
    if unknown:
        raise ValueError(f"not an assay of the checkpoint: {unknown}")
    gid = np.array([assays.index(a) for a in names])
    pred = np.zeros(len(m), np.float32)
    for g in np.unique(gid):  # one assay per batch: every variant has the wild type's length
        rows = np.flatnonzero(gid == g)
        for ix in (rows[a : a + batch_size] for a in range(0, len(rows), batch_size)):
            x = torch.from_numpy(np.stack([fitness_input(m, k) for k in ix])).to(device)
            c = np.zeros((len(ix), lmax), np.uint8)
            for r, k in enumerate(ix):
                s = codes(m.sequence(k))
                c[r, : len(s)] = s
            g_t = torch.from_numpy(gid[ix]).to(device)
            with _autocast(device):
                out = net(x.contiguous(memory_format=torch.channels_last), g_t, torch.from_numpy(c).to(device))
            pred[ix] = out.float().cpu().numpy()
    return pd.DataFrame({"id": m.rows.id, "assay": names, "score": pred})


@torch.no_grad()
def retron(m: Maps, checkpoint, device: str | None = None, batch_size: int = 64):
    """Task 4: predicted msDNA production (inverse-normal scoreINT scale) of Retron-Eco1 ncRNA variants."""
    device = device or checkpoints.default_device()
    net, _ = checkpoints.load(checkpoint, device)
    pred = []
    for ix in _batches(len(m), batch_size):
        xs, cs = zip(*(retron_input(m, k) for k in ix), strict=True)
        x = torch.from_numpy(np.stack(xs)).to(device)
        x = with_sequence(x, torch.from_numpy(np.stack(cs)).to(device)).contiguous(memory_format=torch.channels_last)
        with _autocast(device):
            pred.append(net(x).float().cpu().numpy())
    return pd.DataFrame({"id": m.rows.id, "score": np.concatenate(pred) if pred else []})
