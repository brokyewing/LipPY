"""
LipPhonemeModel — video modeli (Conv3D + ResNetSiLU + Conformer + CTC head).

Eğitim notebook'undaki (kaggle/train-video-model.ipynb) sınıfla birebir uyumludur.
dim_ff inference.py registry'den geçirilir (video modeli = 2048).
best_model.pt checkpoint'i doğrudan yüklenebilir.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchaudio.models import Conformer


# Mimari sabitleri (v1b notebook ile aynı)
D_MODEL = 512
ENC_LAYERS = 6
NHEAD = 8
DIM_FF = 1024
DROPOUT = 0.1


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class BasicBlockSiLU(nn.Module):
    """Eğitim notebook'undaki ile birebir uyumlu — attribute adı 'ds' (NOT 'downsample').

    Checkpoint state_dict key'leri: trunk.layerX.0.ds.0.weight, trunk.layerX.0.ds.1.weight, ...
    'downsample' adı kullanılırsa state_dict eşleşmez.
    """
    expansion = 1

    def __init__(self, ic: int, oc: int, stride: int = 1, ds: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(oc)
        self.conv2 = nn.Conv2d(oc, oc, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(oc)
        self.act = SiLU()
        self.ds = ds  # eğitim notebook'unda 'ds' adı kullanıldı

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.ds is None else self.ds(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class ResNetSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = self._ml(64, 64, 2, 1)
        self.layer2 = self._ml(64, 128, 2, 2)
        self.layer3 = self._ml(128, 256, 2, 2)
        self.layer4 = self._ml(256, 512, 2, 2)

    def _ml(self, ic: int, oc: int, n: int, stride: int) -> nn.Sequential:
        ds = None
        if stride != 1 or ic != oc:
            ds = nn.Sequential(
                nn.Conv2d(ic, oc, 1, stride=stride, bias=False),
                nn.BatchNorm2d(oc),
            )
        layers: list[nn.Module] = [BasicBlockSiLU(ic, oc, stride, ds)]
        layers.extend(BasicBlockSiLU(oc, oc) for _ in range(n - 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class LipPhonemeModel(nn.Module):
    """Video → fonem log-probs.

    Girdi: (B, T, 88, 88) grayscale, normalize edilmiş.
    Çıktı: (T', B, num_phonemes) log-softmax.
    """

    def __init__(
        self,
        num_phonemes: int,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        enc_layers: int = ENC_LAYERS,
        dim_ff: int = DIM_FF,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.d_model = d_model

        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            SiLU(),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.trunk = ResNetSiLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, d_model) if d_model != 512 else nn.Identity()
        self.conformer = Conformer(
            input_dim=d_model,
            num_heads=nhead,
            ffn_dim=dim_ff,
            num_layers=enc_layers,
            depthwise_conv_kernel_size=31,
            dropout=dropout,
        )
        self.ctc_fc = nn.Linear(d_model, num_phonemes)

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, T, H, W) → (B, 1, T, H, W) → frontend3D → (B, 64, T, H', W')
        x = x.unsqueeze(1)
        x = self.frontend3D(x)
        B, C, T, H, W = x.shape
        # (B, T, C, H, W) → (B*T, C, H, W) → trunk → avg_pool → (B*T, 512)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.avg_pool(self.trunk(x)).view(B, T, -1)
        x = self.proj(x)
        if input_lengths is None:
            input_lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
        x, _ = self.conformer(x, lengths=input_lengths)
        logits = self.ctc_fc(x).log_softmax(-1)
        return logits.permute(1, 0, 2)  # (T, B, V)
