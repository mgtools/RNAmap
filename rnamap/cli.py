"""rnamap: download the benchmarks, compute the five maps of your RNAs, and run the four tasks' models.

    rnamap download   [--task ...] [--out data]          the Zenodo benchmarks, checked and unzipped
    rnamap featurize  rnas.fa --out rnas                 rnas.npz + rnas.tsv: the five maps of each sequence
    rnamap classify   rnas                               Task 1: ncRNA class (16 classes)
    rnamap fold       rnas                               Task 2: secondary structure (dot-bracket)
    rnamap variants   wt.fa mutants.tsv --out v.fa       variant sequences from mutant strings (e.g. C10A,U56C)
    rnamap zeroshot   wt mutants.tsv                     Task 3, zero-shot: coupling score, from the wild type's maps
    rnamap fitness    variants --assay A --checkpoint C  Task 3, supervised (M2), for the 31 RNAGym assays
    rnamap retron     variants --checkpoint C            Task 4: msDNA production of Retron-Eco1 ncRNA variants
    rnamap evaluate   --task T --data data               score a model on a released benchmark

`rnas` names the pair of files `featurize` writes (rnas.npz, rnas.tsv). Tables go to stdout unless --out is given.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ("ncrna_classification", "secondary_structure", "variant_fitness", "retron_msdna")


def write_table(table: pd.DataFrame, out: str | None) -> None:
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, sep="\t", index=False, float_format="%.6g")
        print(f"wrote {out}", file=sys.stderr)
    else:
        table.to_csv(sys.stdout, sep="\t", index=False, float_format="%.6g")


def read_variants(path: str, wt: str) -> pd.DataFrame:
    """A table with a `mutant` column (1-based, e.g. C10A,U56C; empty for the wild type) or a `sequence` column."""
    from .zeroshot import apply_mutant

    t = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "sequence" not in t:
        if "mutant" not in t:
            raise SystemExit(f"{path}: needs a 'mutant' or a 'sequence' column")
        t["sequence"] = [apply_mutant(wt, m) for m in t.mutant]
    if "id" not in t:
        t.insert(0, "id", t.mutant if "mutant" in t else [f"v{k}" for k in range(len(t))])
    return t


def cmd_download(a) -> None:
    from .download import download

    download(a.task or list(TASKS), Path(a.out), a.keep_zip)


def cmd_featurize(a) -> None:
    from .features import featurize, read_fasta

    ids, seqs = read_fasta(a.fasta)
    m = featurize(ids, seqs, a.contrafold, a.eternafold, a.eternafold_params, a.device, a.workers)
    m.save(a.out)
    print(f"wrote the maps of {len(m)} RNAs to {a.out}.npz and {a.out}.tsv", file=sys.stderr)


def cmd_classify(a) -> None:
    from .maps import Maps
    from .predict import classify

    table, P, classes = classify(Maps.load(a.maps), a.checkpoint or "ncrna_classification", a.device, a.batch_size)
    if a.probabilities:
        table = pd.concat([table, pd.DataFrame(P, columns=classes)], axis=1)
    write_table(table, a.out)


def cmd_fold(a) -> None:
    from .maps import Maps
    from .predict import fold

    m = Maps.load(a.maps)
    table, probs = fold(m, a.checkpoint or "secondary_structure", a.device, a.threshold)
    if a.bpp:
        np.savez_compressed(a.bpp, **{str(i): P.astype(np.float16) for i, P in zip(m.rows.id, probs, strict=True)})
        print(f"wrote pair probabilities to {a.bpp}", file=sys.stderr)
    write_table(table, a.out)


def cmd_variants(a) -> None:
    from .features import read_fasta

    _, seqs = read_fasta(a.wild_type)
    t = read_variants(a.mutants, seqs[0].upper().replace("T", "U"))
    Path(a.out).write_text("".join(f">{i}\n{s}\n" for i, s in zip(t.id, t.sequence, strict=True)))
    print(f"wrote {len(t)} variants to {a.out}", file=sys.stderr)


def cmd_zeroshot(a) -> None:
    from .maps import Maps
    from .zeroshot import coupling_score

    wt_maps = Maps.load(a.wild_type)
    k = 0 if a.wt_id is None else wt_maps.rows.id.tolist().index(a.wt_id)
    wt = wt_maps.sequence(k)
    R = wt_maps.image(k).mean(0)  # the mean of the five channels
    t = read_variants(a.mutants, wt)
    lam = math.inf if a.lam == "inf" else float(a.lam)
    t["score"] = coupling_score(wt, t.sequence.tolist(), R, lam)
    write_table(t.drop(columns=["sequence"]), a.out)


def cmd_fitness(a) -> None:
    from .maps import Maps
    from .predict import fitness

    write_table(fitness(Maps.load(a.maps), a.assay, a.checkpoint, a.device, a.batch_size), a.out)


def cmd_retron(a) -> None:
    from .maps import Maps
    from .predict import retron

    write_table(retron(Maps.load(a.maps), a.checkpoint, a.device, a.batch_size), a.out)


def cmd_evaluate(a) -> None:
    from . import evaluate

    if a.task == "ncrna_classification":
        print(json.dumps(evaluate.classification(a.data, a.checkpoint or "ncrna_classification", a.device), indent=2))
    elif a.task == "secondary_structure":
        print(evaluate.secondary_structure(a.data, a.checkpoint or "secondary_structure", a.device).to_string(
            index=False, float_format="{:.4f}".format))  # fmt: skip
    elif a.task == "variant_fitness":
        if a.checkpoint:
            per, summary = evaluate.supervised_fitness(a.data, a.checkpoint, a.fold, a.device)
        else:
            per, summary = evaluate.zeroshot_fitness(a.data, evaluate.lam_value(a.lam))
        if a.out:
            write_table(per, a.out)
        print(summary.to_string(index=False, float_format="{:.4f}".format))
    else:
        if not a.checkpoint:
            raise SystemExit("--checkpoint is required for retron_msdna (no released weights)")
        print(json.dumps(evaluate.retron(a.data, a.checkpoint, a.device), indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="rnamap", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(fn=fn)
        return p

    def model_args(p, checkpoint_required=False, batch=16):
        p.add_argument("--checkpoint", required=checkpoint_required,
                       help="a checkpoint file" + ("" if checkpoint_required else " (default: the released one)"))  # fmt: skip
        p.add_argument("--device", help="cuda or cpu (default: cuda if available)")
        p.add_argument("--batch-size", type=int, default=batch)
        p.add_argument("--out", help="output TSV (default: stdout)")

    p = add("download", cmd_download, "download, check and unzip the benchmarks from Zenodo")
    p.add_argument("--task", nargs="+", choices=TASKS, help="default: all four")
    p.add_argument("--out", default="data")
    p.add_argument("--keep-zip", action="store_true")

    p = add("featurize", cmd_featurize, "the five base-pair maps of the sequences of a FASTA file")
    p.add_argument("fasta")
    p.add_argument("--out", required=True, help="output prefix: writes <out>.npz and <out>.tsv")
    p.add_argument("--contrafold", help="CONTRAfold 2.02 binary (default: $CONTRAFOLD, else `contrafold` on the PATH)")
    p.add_argument(
        "--eternafold",
        help="EternaFold binary (default: $ETERNAFOLD or $ETERNAFOLD_PATH, else `eternafold` on the PATH)",
    )
    p.add_argument(
        "--eternafold-params",
        help="EternaFoldParams.v1 (default: $ETERNAFOLD_PARAMS or $ETERNAFOLD_PARAMETERS, else the one bioconda's "
        "eternafold ships)",
    )
    p.add_argument("--device", default="cpu", help="device for RibonanzaNet-SS and ERNIE-RNA")
    p.add_argument("--workers", type=int, default=4, help="processes for RNAfold, CONTRAfold and EternaFold")

    p = add("classify", cmd_classify, "Task 1: predict the ncRNA class")
    p.add_argument("maps", help="prefix of featurize's output")
    model_args(p)
    p.add_argument("--probabilities", action="store_true", help="also write every class's probability")

    p = add("fold", cmd_fold, "Task 2: predict the secondary structure")
    p.add_argument("maps")
    model_args(p, batch=1)
    p.add_argument("--threshold", type=float, help="decoding threshold (default: the checkpoint's, 0.2)")
    p.add_argument("--bpp", help="also write each RNA's (L, L) pair probabilities to this .npz")

    p = add("variants", cmd_variants, "write variant sequences from a wild type and mutant strings")
    p.add_argument("wild_type", help="FASTA with the wild type (its first sequence)")
    p.add_argument("mutants", help="TSV with a 'mutant' column, e.g. C10A,U56C (1-based)")
    p.add_argument("--out", required=True, help="output FASTA")

    p = add("zeroshot", cmd_zeroshot, "Task 3, zero-shot: the coupling score of variants against the wild type's map")
    p.add_argument("wild_type", help="prefix of featurize's output for the wild type")
    p.add_argument("mutants", help="TSV with a 'mutant' (e.g. C10A,U56C) or a 'sequence' column; optional 'id'")
    p.add_argument("--wt-id", help="which RNA of the wild-type files (default: the first)")
    p.add_argument(
        "--lam", default="1", help="weight of a broken pair against a substitution (paper: 1; 'inf': map only)"
    )
    p.add_argument("--out", help="output TSV (default: stdout)")

    p = add("fitness", cmd_fitness, "Task 3, supervised (M2): score variants of one of the 31 RNAGym assays")
    p.add_argument("maps", help="prefix of featurize's output for the variants")
    p.add_argument("--assay", required=True, help="the RNAGym assay, e.g. Andreasson_2020_ribozyme")
    model_args(p, checkpoint_required=True, batch=64)

    p = add("retron", cmd_retron, "Task 4: predict msDNA production of Retron-Eco1 ncRNA variants")
    p.add_argument("maps")
    model_args(p, checkpoint_required=True, batch=64)

    p = add("evaluate", cmd_evaluate, "score a model on a released benchmark, as in the paper")
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--data", default="data", help="where `rnamap download` put the benchmarks")
    p.add_argument("--checkpoint", help="Tasks 1-2: default the released one; Task 3: a supervised one (else "
                   "zero-shot); Task 4: required")  # fmt: skip
    p.add_argument("--fold", type=int, default=0, help="Task 3 supervised: the cross-validation fold it was trained on")
    p.add_argument("--lam", default="1", help="Task 3 zero-shot: lambda of the coupling score")
    p.add_argument("--device")
    p.add_argument("--out", help="Task 3: per-assay table")

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
