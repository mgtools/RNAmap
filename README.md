# RNAmap

RNAmap learns ncRNA representations from multi-channel base-pairing maps. For each RNA, the base-pair probability
matrices of five structure predictors are stacked as the channels of an L × L image: RNAfold, CONTRAfold, EternaFold,
RibonanzaNet-SS and ERNIE-RNA. The eight one-hot sequence channels can be added to them, and one vision network reads
the stack.

This repository holds the inference code for the paper's four tasks, the released checkpoints, a script that
downloads the benchmarks, and examples you can run on your own RNAs. The training code is not included.

| Task                           | What it predicts                               | Model                                                          | Weights    |
| ------------------------------ | ---------------------------------------------- | -------------------------------------------------------------- | ---------- |
| 1. ncRNA classification        | one of 16 ncRNA classes                        | RNAmap-full (13 channels), ResNet-34, RNAcentral pretraining   | released   |
| 2. Secondary structure         | base pairs (dot-bracket)                       | RNAmap-str (5 maps), local pair CNN, nested canonical decoding | released   |
| 3. Variant fitness, zero-shot  | fitness ranking of variants of any RNA         | coupling score from the wild type's map (no training)          | not needed |
| 3. Variant fitness, supervised | fitness of variants of the 31 RNAGym assays    | RNAmap-str, one pooled CNN for all assays                      | code only  |
| 4. Retron msDNA production     | msDNA production of Retron-Eco1 ncRNA variants | RNAmap-full, ResNet-34                                         | code only  |

## Install

With Conda installed, clone the repository and install everything with one command from its root:

```bash
git clone https://github.com/artcmd/RNAmap.git && cd RNAmap
conda env create -f environment.yml && conda activate rnamap
```

This creates a Python 3.12 environment, installs CONTRAfold 2.02 and EternaFold 1.3.1 from Bioconda (including
EternaFold's parameters), and installs RNAmap with all Python dependencies from `pyproject.toml`, including ViennaRNA
and MultiMolecule. The folding tools and parameters are found automatically.

In a new terminal, run `conda activate rnamap` before using `rnamap`. Check the installation with:

```bash
rnamap --help
```

Model weights download on first use: RibonanzaNet-SS and ERNIE-RNA from Hugging Face, and the released RNAmap
checkpoints from GitHub. Benchmarks are downloaded separately with `rnamap download` below.

For an existing Python environment (Python >= 3.10), `pip install -e .` installs RNAmap and its Python dependencies.
CONTRAfold and EternaFold still need to be installed separately. Custom installations can override the detected paths
with `--contrafold`, `--eternafold`, and `--eternafold-params`, or `$CONTRAFOLD`, `$ETERNAFOLD`, and `$ETERNAFOLD_PARAMS`.
Bioconda's `$ETERNAFOLD_PATH` and `$ETERNAFOLD_PARAMETERS` variables are also supported. Explicit command-line paths take
precedence over environment variables, then automatic discovery.

## Data

The four benchmarks are on Zenodo ([record 23967586](https://zenodo.org/records/23967586)), one zip per task. The
following command downloads them, checks each file against Zenodo's MD5 and unzips it into `data/<task>/`:

```bash
rnamap download --out data                                  # all four, 4.7 GB zipped
rnamap download --task secondary_structure retron_msdna     # or only some
```

Each task directory holds `labels.tsv` (one row per RNA, every split of the paper as columns) and one `<split>.npz` of
maps per split. See [Map format](#map-format) for the format and the Zenodo README for the columns.

## Run it on your own RNAs

The `examples/` directory holds small inputs, and `examples/expected/` holds the outputs these commands produce on a
GPU. On the CPU the networks run in fp32 instead of bf16, so the class probabilities can differ in the third decimal.

**1. Compute the five maps.**

```bash
rnamap featurize examples/ncrna.fa --out work/ncrna --device cuda
```

This writes `work/ncrna.npz` (the maps) and `work/ncrna.tsv` (id, length, sequence). It takes about 20 s for the six
example RNAs on a GPU. Sequences may hold A, C, G, U (T is read as U), up to 1,022 nt.

**2. Task 1: classify.**

```bash
rnamap classify work/ncrna --out work/ncrna_classes.tsv
```

The output has the id, the predicted class and its probability; add `--probabilities` for all 16. The classes are
5.8S rRNA, 5S rRNA, CD-box, HACA-box, IRES, group I and II introns, Y RNA, circRNA, leader, lncRNA, miRNA, piRNA,
riboswitch, scaRNA and tRNA. Because the model was trained with label smoothing 0.1, a confident call scores about 0.9.

**3. Task 2: predict the secondary structure.**

```bash
rnamap fold work/ncrna --out work/ncrna_structures.tsv --bpp work/ncrna_bpp.npz
```

The output has one dot-bracket structure per RNA: nested canonical pairs, decoded at the threshold tuned on the held-out
validation families (0.2). `--bpp` also saves each RNA's L × L pair probabilities.

**4. Task 3, zero-shot: rank the variants of a wild type.** Only the wild type needs maps:

```bash
rnamap featurize examples/hammerhead_wt.fa --out work/hammerhead_wt
rnamap zeroshot work/hammerhead_wt examples/hammerhead_mutants.tsv --out work/hammerhead_scores.tsv
```

`hammerhead_mutants.tsv` lists the variants as mutant strings: 1-based, with the wild-type base first, e.g.
`C10A,U56C`. A `sequence` column works instead. The score is s₁ = −n_mut − Δ, where Δ is the expected number of
wild-type base pairs a variant breaks. Higher means predicted fitter. On these 302 variants of the RNAGym hammerhead
assay, its Spearman ρ with the measured `dms_score` is 0.15; counting mutations alone gives 0.02. Use `--lam inf` for
the map term alone.

**5. Task 3, supervised, and Task 4 need a checkpoint.** No weights are released for these two; `--checkpoint` takes
a file in the [checkpoint format](#checkpoint-format).

```bash
rnamap variants examples/hammerhead_wt.fa examples/hammerhead_mutants.tsv --out work/hh_variants.fa
rnamap featurize work/hh_variants.fa --out work/hh_variants
rnamap fitness work/hh_variants --assay Roberts_2023_hh_ribozyme --checkpoint fitness_m2.pt

rnamap featurize examples/retron_variants.fa --out work/retron
rnamap retron work/retron --checkpoint retron.pt
```

The supervised fitness model knows only the 31 RNAGym assays (`data/variant_fitness/assays.tsv`), and every variant must
have its assay's length. Its score is only comparable within an assay. The retron model reads Retron-Eco1 ncRNA
variants of up to 192 nt.

**From Python:**

```python
from rnamap.maps import Maps, Split
from rnamap import predict

m = Maps.load("work/ncrna")  # or Split("data", "secondary_structure", "test_new")
table, probs, class_names = predict.classify(m)
structures, bpp = predict.fold(m)  # dot-brackets, (L, L) probabilities
```

## Reproduce the paper's test numbers

```bash
rnamap evaluate --task ncrna_classification --data data
rnamap evaluate --task secondary_structure  --data data
rnamap evaluate --task variant_fitness      --data data      # zero-shot; ~6 GB of RAM, under 2 min
```

Each released checkpoint is the best of the paper model's three seeds on its test set: seed 2 for Task 1 and seed 1
for Task 2. Its scores therefore sit slightly above the paper's mean over seeds:

| Task | Split                              | Metric                                | This checkpoint | Paper (mean over seeds)                           |
| ---- | ---------------------------------- | ------------------------------------- | --------------- | ------------------------------------------------- |
| 1    | `test` (10,016)                    | accuracy                              | 0.9785          | 0.9781 ± 0.0006 (3 seeds: 0.9774, 0.9784, 0.9785) |
| 2    | `test_new`, bpRNA-new (5,401)      | F-value                               | 0.737           | 0.732 (3 seeds: 0.730, 0.737, 0.728)              |
| 2    | `test_ts0`, TS0 (1,305)            | F-value                               | 0.670           | 0.672                                             |
| 2    | `test_pdb`, primary (83)           | F-value                               | 0.657           | 0.642                                             |
| 3    | all 856,628 variants, zero-shot s₁ | signed Spearman, macro over 3 classes | 0.203           | 0.203                                             |

Task 3's zero-shot score has no seeds; its per-class values are ribozyme 0.202, tRNA 0.485 and aptamer −0.079. On the
benchmark it scores each variant against a wild-type proxy, the per-pair median map of the assay's single mutants, as in
the paper. On a GPU (bf16) the numbers can move in the fourth decimal between runs.

`rnamap evaluate --task variant_fitness --checkpoint C --fold k` scores a supervised checkpoint on the test variants of
cross-validation fold k. `rnamap evaluate --task retron_msdna --checkpoint C` reports Pearson R on the 342 test variants.

## Map format

A set of RNAs is a `.npz` of maps plus a table (`.tsv`, with at least `id` and `sequence`). The `.npz` holds plain
arrays, with no pickles:

| Array      | Shape, dtype     | Content                                                               |
| ---------- | ---------------- | --------------------------------------------------------------------- |
| `id`       | (n,) str         | RNA id, as in the table                                               |
| `length`   | (n,) int32       | sequence length L                                                     |
| `channels` | (5,) str         | `RNAfold`, `CONTRAfold`, `EternaFold`, `RibonanzaNet-SS`, `ERNIE-RNA` |
| `offsets`  | (n + 1,) int64   | the entries of RNA k are `offsets[k] : offsets[k + 1]`                |
| `i`, `j`   | (nnz,) uint16    | 0-based positions, i < j                                              |
| `p`        | (nnz, 5) float16 | pair probability, one column per channel                              |

Each map is symmetric, so only i < j is stored, from p ≥ 10⁻³ (ERNIE-RNA: 2 × 10⁻³). Each network's input is built
from these maps as the paper's model was trained (`rnamap/inputs.py`):

- **Task 1:** a 512 × 512 canvas. Longer RNAs are shrunk onto it, each pixel keeping its largest probability.
- **Task 2:** L × L.
- **Task 3:** entries below 0.01 dropped.
- **Task 4:** a 192 × 192 canvas.

The 8 sequence channels are the one-hot base at i (A, C, G, U; constant along each row), then at j.

`rnamap featurize` reproduces the released maps exactly for RNAfold, CONTRAfold, EternaFold and ERNIE-RNA.
RibonanzaNet-SS comes from MultiMolecule's port of the official weights, which is close to the released maps but not
bit for bit (correlation 0.98–0.996 on the example RNAs). The difference changes no Task 1 or Task 2 prediction of
the examples, and moves bpRNA-new F-values by less than 0.001.

## Checkpoint format

A checkpoint is a `torch.save` dict that loads with `weights_only=True`:

| Key          | Content                                                                                                         |
| ------------ | --------------------------------------------------------------------------------------------------------------- |
| `model`      | the network's class name in `rnamap/models.py`: `Classifier`, `PairNet`, `FitnessRegressor` or `Regressor`      |
| `kwargs`     | its constructor arguments                                                                                       |
| `state_dict` | the weights                                                                                                     |
| task fields  | `classes` (Task 1); `threshold` (Task 2); `assays`, the 31 assay names in id order, as in `assays.tsv` (Task 3) |

The released files are GitHub release assets, downloaded to `~/.cache/rnamap` on first use and checked by MD5. Set
`$RNAMAP_CACHE` to put them elsewhere, or `--checkpoint` to use a local file.

| File                             | Task | Size   |
| -------------------------------- | ---- | ------ |
| `rnamap_ncrna_classification.pt` | 1    | 86 MB  |
| `rnamap_secondary_structure.pt`  | 2    | 2.4 MB |

## Layout

```
rnamap/
  cli.py          the `rnamap` command
  download.py     Zenodo download, MD5 check, unzip
  features.py     the five maps of new sequences
  maps.py         the map format, benchmark splits, dot-bracket helpers
  inputs.py       each task's network input
  models.py       the four networks
  structure.py    nested canonical decoding, base-pair scoring
  zeroshot.py     the coupling score
  predict.py      inference for each task
  checkpoints.py  released checkpoints: fetch, check, load
  evaluate.py     the paper's metric for each task
examples/         inputs for the commands above, and their outputs in expected/
tests/            CPU tests: pytest
```

## Licence and citation

The code is under the MIT licence (`LICENSE`). The benchmark labels come from the sources listed in the Zenodo README,
whose terms apply. RibonanzaNet-SS and ERNIE-RNA are downloaded at run time from MultiMolecule, under their own
licences. Please cite those works together with the RNAmap paper (citation to follow).
