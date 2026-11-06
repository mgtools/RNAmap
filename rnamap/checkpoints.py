"""Released checkpoints: where they live, how to fetch them, how to build their networks.

A checkpoint is a plain `torch.save` dict (loadable with `weights_only=True`):

    model       name of the network in `rnamap.models` (Classifier, PairNet, FitnessRegressor, Regressor)
    kwargs      its constructor arguments
    state_dict  the weights
    ...         task metadata: `classes` (Task 1), `threshold` (Task 2), `assays` (Task 3)

Released (GitHub release assets): Task 1 and Task 2. Tasks 3 and 4 ship as code: `--checkpoint` takes a file in the
format above (`FitnessRegressor` with `assays`, the 31 RNAGym assay names in id order; `Regressor`).
"""

import hashlib
import os
import urllib.request
from pathlib import Path

import torch

from . import models

RELEASE_URL = os.environ.get("RNAMAP_RELEASE_URL", "https://github.com/artcmd/RNAmap/releases/download/v1.0.0")
CACHE = Path(os.environ.get("RNAMAP_CACHE", Path.home() / ".cache" / "rnamap"))

RELEASED = {  # name -> (file, md5)
    "ncrna_classification": ("rnamap_ncrna_classification.pt", "9465cc5afeb9c5b063cc391139e97909"),
    "secondary_structure": ("rnamap_secondary_structure.pt", "72fca6597e45d55dfc52906d32d3adde"),
}


def md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def fetch(name: str) -> Path:
    """The local path of released checkpoint `name`, downloaded to $RNAMAP_CACHE (~/.cache/rnamap) on first use."""
    file, digest = RELEASED[name]
    path = CACHE / file
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        url = f"{RELEASE_URL}/{file}"
        print(f"downloading {url}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    if digest != "TODO" and md5(path) != digest:
        raise RuntimeError(f"{path}: MD5 mismatch; delete it and retry")
    return path


def load(path_or_name, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """The network in eval mode on `device`, and the checkpoint's other fields."""
    path = Path(path_or_name) if Path(path_or_name).exists() else fetch(path_or_name)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = getattr(models, ck["model"])(**ck.get("kwargs", {}))
    net.load_state_dict(ck["state_dict"])
    meta = {k: v for k, v in ck.items() if k != "state_dict"}
    return net.to(device).eval(), meta


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
