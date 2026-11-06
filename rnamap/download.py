"""Download the RNAmap benchmarks from Zenodo (record 23967586), check their MD5 and unzip them.

data/
  ncrna_classification/   labels.tsv, train.npz, test.npz                          (0.7 GB zipped)
  secondary_structure/    labels.tsv, train/val/test_ts0/test_new/test_pdb.npz      (84 MB)
  variant_fitness/        labels.tsv, assays.tsv, fold0-4.npz                       (3.9 GB)
  retron_msdna/           labels.tsv, train/val/test.npz                            (15 MB)
"""

import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

RECORD = "23967586"
API = f"https://zenodo.org/api/records/{RECORD}"
TASKS = ("ncrna_classification", "secondary_structure", "variant_fitness", "retron_msdna")


def record_files() -> dict[str, dict]:
    """{file name: {"url", "md5", "size"}} of the Zenodo record."""
    req = urllib.request.Request(API, headers={"Accept": "application/json", "User-Agent": "rnamap-download"})
    with urllib.request.urlopen(req, timeout=60) as r:
        rec = json.load(r)
    return {
        f["key"]: {"url": f["links"]["self"], "md5": f["checksum"].removeprefix("md5:"), "size": f["size"]}
        for f in rec["files"]
    }


def md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def fetch(url: str, dest: Path, size: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "rnamap-download"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        done = 0
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {dest.name}: {done / 1e6:,.0f} / {size / 1e6:,.0f} MB", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    tmp.rename(dest)


def download(tasks: list[str], out: Path, keep_zip: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    files = record_files()
    for task in tasks:
        name = f"{task}.zip"
        info = files[name]
        if (out / task / "labels.tsv").exists():
            print(f"{task}: already in {out / task}, skipped")
            continue
        z = out / name
        if not (z.exists() and md5(z) == info["md5"]):
            fetch(info["url"], z, info["size"])
        if md5(z) != info["md5"]:
            raise RuntimeError(f"{z}: MD5 mismatch with Zenodo; delete it and retry")
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():  # members are <task>/<file>; refuse anything else
                p = Path(m.filename)
                if p.is_absolute() or ".." in p.parts or p.parts[0] != task:
                    raise RuntimeError(f"{z}: unexpected member {m.filename}")
            tmp = out / f".{task}.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            zf.extractall(tmp)
            (tmp / task).rename(out / task)
            tmp.rmdir()
        if not keep_zip:
            z.unlink()
        print(f"{task}: {out / task}")
