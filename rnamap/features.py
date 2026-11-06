"""The five base-pair maps of new sequences, computed as for the released benchmarks.

    RNAfold          ViennaRNA (pip `ViennaRNA`), partition function with default settings
    CONTRAfold       CONTRAfold 2.02 (bioconda `contrafold`): `contrafold predict --posteriors 0.001`
    EternaFold       EternaFold 1.3.1 (bioconda `eternafold`): the same, with `--params EternaFoldParams.v1`
    RibonanzaNet-SS  MultiMolecule's port of the official weights (`multimolecule/ribonanzanet-ss`), sigmoid of the
                     pair logits
    ERNIE-RNA        ERNIE-RNA zero-shot (`multimolecule/ernierna`): the attention of head 5 in layers 9, 10 and 11
                     (0-based), each symmetrised, averaged, and calibrated as sigmoid(-6.8036 + 10.1053 x)

Pairs are kept from p >= 1e-3 (ERNIE-RNA: 2e-3, and only |i - j| >= 4). These are the settings used to make the
external PDB test set; on bpRNA they reproduce the benchmark maps exactly except RibonanzaNet-SS, whose MultiMolecule
port changes bpRNA-new F-values by less than 0.001. ERNIE-RNA reads at most 1,022 nt.

`folding_tools` locates CONTRAfold, EternaFold and EternaFold's parameters: an explicit path, else $CONTRAFOLD,
$ETERNAFOLD (or bioconda's $ETERNAFOLD_PATH) and $ETERNAFOLD_PARAMS (or $ETERNAFOLD_PARAMETERS), else the `contrafold`
and `eternafold` commands of this Python's environment or the PATH (as the conda environment of `environment.yml`
installs them) and the parameters bioconda's `eternafold` ships.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .maps import Maps, from_dense

ERNIE_HEADS = ((9, 5), (10, 5), (11, 5))
ERNIE_A, ERNIE_B = -6.8036, 10.1053
ERNIE_MIN_SEP = 4
MAX_LEN = 1022


def rnafold(seq: str) -> np.ndarray:
    import RNA

    fc = RNA.fold_compound(seq)
    fc.pf()
    b = np.array(fc.bpp(), dtype=np.float32)[1:, 1:]
    return b + b.T


def contrafold(seq: str, binary: str, params: str | None = None) -> np.ndarray:
    """CONTRAfold's (or, with EternaFold's binary and `params`, EternaFold's) posterior pair probabilities."""
    L = len(seq)
    P = np.zeros((L, L), np.float32)
    with tempfile.TemporaryDirectory() as t:
        Path(t, "x.fa").write_text(f">x\n{seq}\n")
        cmd = [binary, "predict", f"{t}/x.fa", "--posteriors", "0.001", f"{t}/p"] + (
            ["--params", params] if params else []
        )
        subprocess.run(cmd, check=True, capture_output=True)
        for line in Path(t, "p").read_text().splitlines():
            f = line.split()
            for x in f[2:]:
                j, p = x.split(":")
                P[int(f[0]) - 1, int(j) - 1] = float(p)
    return np.maximum(P, P.T)


class Networks:
    """RibonanzaNet-SS and ERNIE-RNA through MultiMolecule, loaded once, run one RNA at a time in fp32."""

    def __init__(self, device: str = "cpu"):
        from multimolecule import ErnieRnaModel, RibonanzaNetForSecondaryStructurePrediction, RnaTokenizer

        self.device = device
        load = {
            "rnet": (RibonanzaNetForSecondaryStructurePrediction, "multimolecule/ribonanzanet-ss"),
            "ernie": (ErnieRnaModel, "multimolecule/ernierna"),
        }
        self.models = {k: cls.from_pretrained(name).to(device).eval() for k, (cls, name) in load.items()}
        self.tok = {k: RnaTokenizer.from_pretrained(name) for k, (_, name) in load.items()}

    def _inputs(self, key: str, seq: str) -> dict:
        return {a: b.to(self.device) for a, b in self.tok[key](seq, return_tensors="pt").items()}

    @torch.no_grad()
    def run(self, seq: str) -> tuple[np.ndarray, np.ndarray]:
        L = len(seq)
        lg = self.models["rnet"](**self._inputs("rnet", seq))["logits_ss"][0, :, :, 0]
        rnet = torch.sigmoid(lg.float()).cpu().numpy()
        att = self.models["ernie"](**self._inputs("ernie", seq), output_attentions=True).attentions
        A = torch.stack([att[layer][0, h, 1 : L + 1, 1 : L + 1].float() for layer, h in ERNIE_HEADS])
        x = ((A + A.transpose(1, 2)) / 2).mean(0)
        ernie = torch.sigmoid(ERNIE_A + ERNIE_B * x).cpu().numpy()
        near = np.abs(np.subtract.outer(np.arange(L), np.arange(L))) < ERNIE_MIN_SEP
        ernie[near] = 0
        return rnet, ernie


def clean(seq: str) -> str:
    s = seq.strip().upper().replace("T", "U")
    bad = set(s) - set("ACGU")
    if bad:
        raise ValueError(f"unsupported letters {sorted(bad)}: the predictors read A, C, G, U only")
    return s


ETERNAFOLD_PARAMS = Path("lib", "eternafold-lib", "parameters", "EternaFoldParams.v1")  # under bioconda's env prefix


def _given(explicit: str | None, *variables: str) -> str | None:
    """An explicit path, else the first environment variable that is set."""
    return explicit or next((os.environ[v] for v in variables if os.environ.get(v)), None)


def _command(name: str) -> str | None:
    """`name` in this Python's environment (a conda env run without activating it), else on the PATH."""
    return shutil.which(name, path=os.path.join(sys.prefix, "bin")) or shutil.which(name)


def folding_tools(contrafold_bin: str | None = None, eternafold_bin: str | None = None,
                  eternafold_params: str | None = None) -> tuple[str, str, str]:  # fmt: skip
    """(CONTRAfold binary, EternaFold binary, EternaFold parameters), or a SystemExit naming what is missing.

    A path given explicitly or by an environment variable must exist: it never falls back to automatic discovery.
    """
    cf_given = _given(contrafold_bin, "CONTRAFOLD")
    ef_given = _given(eternafold_bin, "ETERNAFOLD", "ETERNAFOLD_PATH")
    params_given = _given(eternafold_params, "ETERNAFOLD_PARAMS", "ETERNAFOLD_PARAMETERS")
    cf = shutil.which(cf_given) if cf_given else _command("contrafold")
    ef = shutil.which(ef_given) if ef_given else _command("eternafold")
    if params_given:
        params = params_given if Path(params_given).is_file() else None
    else:  # bioconda: <env>/bin/eternafold and <env>/lib/eternafold-lib/parameters/
        beside = [Path(ef).absolute().parent.parent / ETERNAFOLD_PARAMS] if ef else []
        params = next((str(p) for p in [*beside, Path(sys.prefix) / ETERNAFOLD_PARAMS] if p.is_file()), None)
    tools = (
        ("CONTRAfold", cf_given, cf),
        ("EternaFold", ef_given, ef),
        ("EternaFold parameters", params_given, params),
    )
    missing = [f"{name} (not found at {given})" if given else name for name, given, found in tools if not found]
    if missing:
        raise SystemExit(f"Missing folding tools: {', '.join(missing)}. Install them with `conda env create -f "
                         "environment.yml`, or pass --contrafold, --eternafold and --eternafold-params.")  # fmt: skip
    return cf, ef, params


def featurize(ids: list[str], seqs: list[str], contrafold_bin: str | None = None, eternafold_bin: str | None = None,
              eternafold_params: str | None = None, device: str = "cpu", workers: int = 4) -> Maps:  # fmt: skip
    """The five maps of each sequence, packed in the benchmark format."""
    contrafold_bin, eternafold_bin, eternafold_params = folding_tools(contrafold_bin, eternafold_bin, eternafold_params)
    seqs = [clean(s) for s in seqs]
    too_long = [i for i, s in zip(ids, seqs, strict=True) if len(s) > MAX_LEN]
    if too_long:
        raise ValueError(f"{len(too_long)} sequences are longer than {MAX_LEN} nt (ERNIE-RNA's limit): {too_long[:3]}")
    with ProcessPoolExecutor(workers) as pool:
        rf = list(pool.map(rnafold, seqs))
        cf = list(pool.map(partial(contrafold, binary=contrafold_bin), seqs))
        ef = list(pool.map(partial(contrafold, binary=eternafold_bin, params=eternafold_params), seqs))
    nets = Networks(device)
    maps = []
    for s, a, b, c in zip(seqs, rf, cf, ef, strict=True):
        rnet, ernie = nets.run(s)
        maps.append(np.stack([a, b, c, rnet, ernie]).astype(np.float32))
    return from_dense(ids, seqs, maps)


def read_fasta(path) -> tuple[list[str], list[str]]:
    """(ids, sequences) of a FASTA file; the id is the header's first word."""
    ids, seqs = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            ids.append(line[1:].split()[0] if line[1:].split() else f"seq{len(ids)}")
            seqs.append("")
        elif line:
            if not ids:
                raise ValueError(f"{path}: sequence before the first header")
            seqs[-1] += line
    return ids, seqs
