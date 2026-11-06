"""CPU tests that need neither the benchmarks, the checkpoints, nor the folding tools."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from rnamap import cli
from rnamap.inputs import canvas_codes, canvas_maps, fitness_canvas
from rnamap.maps import Maps, codes, dotbracket_to_pairs, from_dense, pairs_to_dotbracket
from rnamap.models import Classifier, FitnessRegressor, PairNet, Regressor
from rnamap.structure import accuracy, confusion, nested_pairs
from rnamap.zeroshot import apply_mutant, coupling_score, median_map, n_mutations, pairs_broken


def hairpin(seq="GGGGAAACCCC"):
    """A (5, L, L) map with the stem of a GGGG...CCCC hairpin at 0.9 in every channel."""
    L = len(seq)
    x = np.zeros((5, L, L), np.float32)
    for k in range(4):
        x[:, k, L - 1 - k] = x[:, L - 1 - k, k] = 0.9
    return seq, x


def test_maps_roundtrip(tmp_path):
    seq, x = hairpin()
    x[4, 0, 5] = x[4, 5, 0] = 0.0015  # below ERNIE-RNA's floor: dropped
    m = from_dense(["a"], [seq], [x])
    assert len(m.z["i"]) == 4
    m.save(tmp_path / "h")
    back = Maps.load(tmp_path / "h")
    assert np.allclose(back.image(0), np.where(x >= 0.002, x, 0), atol=1e-3)
    assert back.sequence(0) == seq


def test_dotbracket_roundtrip():
    for db in ["((..((...))..))", "((..[[..))..]]", "...."]:
        p = dotbracket_to_pairs(db)
        assert pairs_to_dotbracket(p[:, 0], p[:, 1], len(db)) == db


def test_nested_decoding_and_scores():
    seq, x = hairpin()
    i, j = nested_pairs(x.mean(0), codes(seq), 0.3)
    assert pairs_to_dotbracket(i, j, len(seq)) == "((((...))))"
    true = dotbracket_to_pairs("((((...))))")
    assert accuracy(*confusion(true, np.stack([i, j], 1), len(seq)))[2] == 1.0
    half = dotbracket_to_pairs("((.......))")
    sen, ppv, f, _ = accuracy(*confusion(true, half, len(seq)))
    assert (sen, ppv) == (0.5, 1.0) and math.isclose(f, 2 / 3)


def test_shrink_keeps_largest():
    L = 1024
    i, j = np.array([0, 1]), np.array([600, 601])  # both land on pixel (0, 300)
    p = np.zeros((2, 5), np.float32)
    p[:, 0] = [0.2, 0.7]
    z = {"id": np.array(["x"]), "length": np.array([L], np.int32), "channels": np.array(cli_channels()),
         "offsets": np.array([0, 2]), "i": i.astype(np.uint16), "j": j.astype(np.uint16), "p": p.astype(np.float16)}  # fmt: skip
    m = Maps(z, pd.DataFrame({"id": ["x"], "sequence": ["A" * L]}))
    x = canvas_maps(m, 0, 512)
    assert x.shape == (5, 512, 512) and np.isclose(x[0, 0, 300], 0.7, atol=1e-3) and x[0, 300, 0] == x[0, 0, 300]
    assert (canvas_codes("ACGU" * 256, 512)[:4] == [1, 3, 1, 3]).all()  # the first nucleotide of each pixel
    assert fitness_canvas(74) == 96


def cli_channels():
    from rnamap.maps import CHANNELS

    return CHANNELS


def test_coupling_score():
    wt = "GGGGAAACCCC"
    _, x = hairpin(wt)
    R = x.mean(0)
    variants = [wt, apply_mutant(wt, "G1A,C11U"), apply_mutant(wt, "G1C"), apply_mutant(wt, "A5U")]
    delta = pairs_broken(R, wt, variants)
    assert np.allclose(delta, [0, 0, 0.9, 0])  # G-C -> A-U keeps the pair; G1C breaks it
    assert (n_mutations(wt, variants) == [0, 2, 1, 1]).all()
    assert np.allclose(coupling_score(wt, variants, R, 1.0), [0, -2, -1.9, -1])
    assert np.allclose(coupling_score(wt, variants, R, math.inf), [0, 0, -0.9, 0])
    with pytest.raises(ValueError):
        apply_mutant(wt, "A1G")  # position 1 is G
    assert np.allclose(median_map(np.arange(4.0)[:, None]), [1.5])


@torch.no_grad()
def test_networks_forward():
    assert Classifier(16).eval()(torch.zeros(1, 13, 512, 512)).shape == (1, 16)
    net = PairNet().eval()
    for p in net.parameters():
        p.normal_(0, 0.1)
    z = net(torch.rand(2, 5, 40, 40), torch.tensor([40, 30]))
    assert z.shape == (2, 40, 40) and torch.allclose(z, z.transpose(1, 2))
    assert Regressor().eval()(torch.zeros(2, 13, 192, 192)).shape == (2,)
    f = FitnessRegressor().eval()
    out = f(torch.zeros(3, 5, 96, 96), torch.tensor([0, 0, 0]), torch.zeros(3, 425, dtype=torch.long))
    assert out.shape == (3,)


def test_cli_zeroshot_and_variants(tmp_path, capsys):
    seq, x = hairpin()
    from_dense(["wt"], [seq], [x]).save(tmp_path / "wt")
    (tmp_path / "wt.fa").write_text(f">wt\n{seq}\n")
    pd.DataFrame({"mutant": ["G1A,C11U", "G1C"]}).to_csv(tmp_path / "m.tsv", sep="\t", index=False)
    cli.main(["zeroshot", str(tmp_path / "wt"), str(tmp_path / "m.tsv"), "--out", str(tmp_path / "s.tsv")])
    s = pd.read_csv(tmp_path / "s.tsv", sep="\t")
    assert np.allclose(s.score, [-2, -1.9], atol=1e-3)
    cli.main(["variants", str(tmp_path / "wt.fa"), str(tmp_path / "m.tsv"), "--out", str(tmp_path / "v.fa")])
    assert (tmp_path / "v.fa").read_text().splitlines()[1] == "AGGGAAACCCU"
