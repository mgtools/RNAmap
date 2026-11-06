"""The networks of the four tasks, with the module names of the paper's checkpoints.

Task 1  `Classifier`        torchvision ResNet-34 with a 13-channel stem and an MLP decoder (RNAmap-full)
Task 2  `PairNet`           a 0.6M-weight pair network at L x L resolution: 8 residual blocks of 3 x 3 convs with
                            per-pixel LayerNorm, 33 x 33-pixel receptive field (RNAmap-str, the "local CNN")
Task 3  `FitnessRegressor`  a six-stage CNN with a 3 x 3 grid head, a learned assay embedding and a per-assay
                            additive site table: one pooled model for all 31 RNAGym assays (M2, RNAmap-str)
Task 4  `Regressor`         the ResNet-34 body with the same 3 x 3 grid head (RNAmap-full)
"""

import math

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

N_MAPS = 5
MIN_SEP = 4  # pairs with j - i < 4 are never predicted
MAX_LEN = 512


def with_sequence(x: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Append the 8 one-hot sequence channels to (B, C, S, S) images: [s_i == b], constant along each row, then
    [s_j == b], constant along each column, b in A, C, G, U. `codes` (B, S): 1-4 for A, C, G, U, 0 elsewhere."""
    oh = F.one_hot(codes.long(), 5)[..., 1:].permute(0, 2, 1).to(x.dtype)  # (B, 4, S)
    S = x.shape[-1]
    return torch.cat([x, oh[..., :, None].expand(-1, -1, -1, S), oh[..., None, :].expand(-1, -1, S, -1)], 1)


# Task 1 ---------------------------------------------------------------------------------------------------------------


def Classifier(n_classes: int, in_channels: int = 13, hidden: int = 256, dropout: float = 0.2) -> nn.Module:
    """ResNet-34 on the (13, 512, 512) image, global average pooling, then dropout-linear-ReLU-dropout-linear."""
    net = torchvision.models.resnet34()
    net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(512, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_classes),
    )
    return net


# Task 2 ---------------------------------------------------------------------------------------------------------------


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of each pixel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    """Pre-norm residual block of two convolutions; padded pixels are re-zeroed before each one."""

    def __init__(self, width: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        pad = dilation * (kernel // 2)
        self.norm1, self.conv1 = LayerNorm2d(width), nn.Conv2d(width, width, kernel, padding=pad, dilation=dilation)
        self.norm2, self.conv2 = LayerNorm2d(width), nn.Conv2d(width, width, kernel, padding=pad, dilation=dilation)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x)) * m)
        h = self.conv2(F.gelu(self.norm2(h)) * m)
        return x + h * m


class PairNet(nn.Module):
    """Symmetric pair logits (B, L, L) from the five maps.

    Input features per pixel (17): the five maps, each map's row sums at i and at j (how much each nucleotide pairs
    overall), log(1 + |i - j|) / log(512) and [|i - j| < 4]. A 1 x 1 stem, `depth` residual blocks, a LayerNorm and a
    1 x 1 head; the output is averaged with its transpose.
    """

    def __init__(self, width: int = 64, depth: int = 8, kernel: int = 3, dilations: tuple[int, ...] = (1,)):
        super().__init__()
        self.stem = nn.Conv2d(3 * N_MAPS + 2, width, 1)
        self.blocks = nn.ModuleList(Block(width, kernel, dilations[k % len(dilations)]) for k in range(depth))
        self.head = nn.Sequential(LayerNorm2d(width), nn.Conv2d(width, 1, 1))

    def features(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, n, _ = x.shape
        ar = torch.arange(n, device=x.device)
        valid = ar[None, :] < lengths[:, None].to(x.device)
        m = (valid[:, :, None] & valid[:, None, :])[:, None].to(x.dtype)
        rows = x.sum(-1)  # (B, 5, L): expected number of partners of each nucleotide
        sep = (ar[None, :] - ar[:, None]).abs().to(x.dtype)
        geom = torch.stack([torch.log1p(sep) / math.log(MAX_LEN), (sep < MIN_SEP).to(x.dtype)])
        f = torch.cat([x, rows[..., :, None].expand(-1, -1, -1, n), rows[..., None, :].expand(-1, -1, n, -1),
                       geom.expand(B, -1, -1, -1)], 1)  # fmt: skip
        return f * m, m

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """`x` (B, 5, n, n) zero-padded maps, `lengths` (B,); padded pixels of the output are meaningless."""
        f, m = self.features(x, lengths)
        h = self.stem(f) * m
        for blk in self.blocks:
            h = blk(h, m)
        z = self.head(h)
        return ((z + z.transpose(2, 3)) / 2)[:, 0]


# Tasks 3 and 4 --------------------------------------------------------------------------------------------------------


class SmallCNN(nn.Module):
    """Six stride-2 stages widening 32 -> 256, GroupNorm and SiLU."""

    def __init__(self, in_channels: int, widths=(32, 64, 96, 128, 192, 256)):
        super().__init__()
        layers, c = [], in_channels
        for k, w in enumerate(widths):
            layers += [
                nn.Conv2d(c, w, 5 if k == 0 else 3, stride=2, padding=2 if k == 0 else 1, bias=False),
                nn.GroupNorm(min(8, w), w),
                nn.SiLU(inplace=True),
                nn.Conv2d(w, w, 3, padding=1, bias=False),
                nn.GroupNorm(min(8, w), w),
                nn.SiLU(inplace=True),
            ]
            c = w
        self.body = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ResNetBody(nn.Module):
    """torchvision ResNet-34 up to `layer4`, keeping its module names."""

    def __init__(self, in_channels: int):
        super().__init__()
        net = torchvision.models.resnet34()
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"):
            setattr(self, name, getattr(net, name))
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


def grid_head(c: int, head_channels: int, hidden: int, dropout: float) -> tuple[nn.Module, nn.Module]:
    """The body's grid thinned to `head_channels` maps and pooled to 3 x 3 (so the head knows where in the molecule
    the signal is), then one hidden layer."""
    reduce = nn.Sequential(nn.Conv2d(c, head_channels, 1, bias=False), nn.SiLU(inplace=True), nn.AdaptiveAvgPool2d(3))
    head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(head_channels * 9, hidden), nn.SiLU(inplace=True))
    return reduce, head


class Regressor(nn.Module):
    """Task 4: ResNet-34 body on the (13, 192, 192) image, grid head, one output."""

    def __init__(self, in_channels: int = 13, head_channels: int = 24, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.body = ResNetBody(in_channels)
        self.reduce, head = grid_head(self.body.out_channels, head_channels, hidden, dropout)
        self.head = nn.Sequential(*head, nn.Dropout(dropout), nn.Linear(hidden, 1))  # the paper's checkpoint layout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.reduce(self.body(x)))[:, 0]


class FitnessRegressor(nn.Module):
    """Task 3 (M2): SmallCNN on the five maps, grid head, [hidden, assay embedding] -> hidden -> 1, plus an additive
    term per assay, a weight for every base at every position and an offset: w[g, k, base] + b[g].

    The output is on the scale the model was trained on: each variant's label as an inverse-normal rank among its own
    assay's training labels. Only its order within an assay is meaningful.
    """

    def __init__(self, n_assays: int = 31, lmax: int = 425, in_channels: int = N_MAPS, group_dim: int = 16,
                 head_channels: int = 24, hidden: int = 128, dropout: float = 0.3):  # fmt: skip
        super().__init__()
        self.body = SmallCNN(in_channels)
        self.reduce, self.head = grid_head(self.body.out_channels, head_channels, hidden, dropout)
        self.group = nn.Embedding(n_assays, group_dim)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden + group_dim, hidden), nn.SiLU(inplace=True),
                                 nn.Linear(hidden, 1))  # fmt: skip
        self.add_w = nn.Parameter(torch.zeros(n_assays, lmax, 5))
        self.add_b = nn.Parameter(torch.zeros(n_assays))

    def forward(self, x: torch.Tensor, g: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """`x` (B, 5, S, S) maps on the assay's canvas, `g` (B,) assay ids, `codes` (B, lmax) unshifted base codes."""
        h = torch.cat([self.head(self.reduce(self.body(x))), self.group(g).to(x.dtype)], 1)
        pred = self.out(h)[:, 0].float()
        w = self.add_w[g]  # (B, lmax, 5)
        return pred + torch.gather(w, 2, codes.long()[..., None]).squeeze(2).sum(1) + self.add_b[g]
