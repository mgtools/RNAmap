"""Each task's network input, built from one RNA of a `Maps` set exactly as the paper's models were trained on it.

Task 1  (13, 512, 512)  the five maps and the 8 sequence channels, zero-padded to 512; an RNA longer than 512 nt is
                        shrunk onto the canvas (nucleotide i to pixel floor(512 i / L), each pixel keeping its
                        largest probability, and the first nucleotide of each pixel for the sequence)
Task 2  (5, L, L)       the five maps
Task 3  (5, S, S)       the five maps with entries below 0.01 dropped, on a canvas of S = L + 8 rounded up to 16
Task 4  (13, 192, 192)  the five maps and the 8 sequence channels, zero-padded to 192
"""

import numpy as np

from .maps import CHANNELS, Maps, codes

CLS_SIZE = 512
RETRON_SIZE = 192
FITNESS_MIN_P = 0.01  # the paper's Task 3 models read the maps without entries below 0.01
FITNESS_SHIFT = 8


def canvas_codes(seq: str, size: int) -> np.ndarray:
    """(size,) uint8 base codes on the canvas; a longer sequence keeps the first nucleotide of each pixel."""
    nt, out = codes(seq), np.zeros(size, np.uint8)
    n = len(nt)
    out[: min(n, size)] = nt if n <= size else nt[-(-np.arange(size) * n // size)]
    return out


def canvas_maps(m: Maps, k: int, size: int, min_p: float = 0.0) -> np.ndarray:
    """(5, size, size) float32: RNA k's maps at the top left of the canvas, shrunk onto it if L > size."""
    L = int(m.lengths[k])
    i, j, p = m.pairs(k, min_p)
    x = np.zeros((len(CHANNELS), size, size), np.float32)
    if L > size:
        i, j = i * size // L, j * size // L
    for c in range(len(CHANNELS)):
        np.maximum.at(x[c], (i, j), p[:, c])
        np.maximum.at(x[c], (j, i), p[:, c])
    return x


def classification_input(m: Maps, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Task 1: the (5, 512, 512) maps and the (512,) base codes the sequence channels are drawn from."""
    return canvas_maps(m, k, CLS_SIZE), canvas_codes(m.sequence(k), CLS_SIZE)


def retron_input(m: Maps, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Task 4: the (5, 192, 192) maps and the (192,) base codes."""
    assert m.lengths[k] <= RETRON_SIZE, f"{m.rows.id[k]}: longer than the {RETRON_SIZE}-pixel canvas"
    return canvas_maps(m, k, RETRON_SIZE), canvas_codes(m.sequence(k), RETRON_SIZE)


def fitness_canvas(L: int) -> int:
    return -(-(L + FITNESS_SHIFT) // 16) * 16


def fitness_input(m: Maps, k: int) -> np.ndarray:
    """Task 3: the (5, S, S) maps, entries below 0.01 dropped."""
    return canvas_maps(m, k, fitness_canvas(int(m.lengths[k])), FITNESS_MIN_P)
