"""
LipPhonemeModel — video modeli (Conv3D + ResNetSiLU + Conformer + CTC head).

Eğitim notebook'undaki (kaggle/train-video-model.ipynb) sınıfla birebir uyumludur.
torchaudio.models.Conformer yüklü değilse birebir aynı state_dict anahtarlarına ve
matematiğe sahip saf PyTorch Conformer kullanılır (strict=True ile 320/320 anahtar eşleşir).
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchaudio.models import Conformer  # type: ignore
except ImportError:
    class _FeedForwardModule(nn.Module):
        def __init__(self, input_dim: int, ffn_dim: int, dropout: float = 0.0):
            super().__init__()
            self.sequential = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, ffn_dim, bias=True),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, input_dim, bias=True),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.sequential(x)

    class _ConvolutionModule(nn.Module):
        def __init__(self, input_dim: int, num_channels: int, depthwise_kernel_size: int, dropout: float = 0.0):
            super().__init__()
            self.layer_norm = nn.LayerNorm(input_dim)
            self.sequential = nn.Sequential(
                nn.Conv1d(input_dim, 2 * num_channels, 1, stride=1, padding=0, bias=True),
                nn.GLU(dim=1),
                nn.Conv1d(
                    num_channels,
                    num_channels,
                    depthwise_kernel_size,
                    stride=1,
                    padding=(depthwise_kernel_size - 1) // 2,
                    groups=num_channels,
                    bias=True,
                ),
                nn.BatchNorm1d(num_channels),
                nn.SiLU(),
                nn.Conv1d(num_channels, input_dim, 1, stride=1, padding=0, bias=True),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (T, B, D)
            x = self.layer_norm(x)
            x = x.permute(1, 2, 0)  # (B, D, T)
            x = self.sequential(x)
            return x.permute(2, 0, 1)  # (T, B, D)

    class _ConformerLayer(nn.Module):
        def __init__(
            self,
            input_dim: int,
            ffn_dim: int,
            num_attention_heads: int,
            depthwise_conv_kernel_size: int,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
            self.self_attn_layer_norm = nn.LayerNorm(input_dim)
            self.self_attn = nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
            self.self_attn_dropout = nn.Dropout(dropout)
            self.conv_module = _ConvolutionModule(
                input_dim=input_dim,
                num_channels=input_dim,
                depthwise_kernel_size=depthwise_conv_kernel_size,
                dropout=dropout,
            )
            self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
            self.final_layer_norm = nn.LayerNorm(input_dim)

        def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
            # x: (T, B, D)
            residual = x
            x = self.ffn1(x)
            x = x * 0.5 + residual

            residual = x
            x = self.self_attn_layer_norm(x)
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            x = self.self_attn_dropout(x)
            x = x + residual

            residual = x
            x = self.conv_module(x)
            x = x + residual

            residual = x
            x = self.ffn2(x)
            x = x * 0.5 + residual

            return self.final_layer_norm(x)

    class Conformer(nn.Module):
        def __init__(
            self,
            input_dim: int,
            num_heads: int,
            ffn_dim: int,
            num_layers: int,
            depthwise_conv_kernel_size: int,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.conformer_layers = nn.ModuleList(
                [
                    _ConformerLayer(
                        input_dim=input_dim,
                        ffn_dim=ffn_dim,
                        num_attention_heads=num_heads,
                        depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )

        def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # input: (B, T, D)
            max_len = input.size(1)
            mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
            x = input.transpose(0, 1)  # (T, B, D)
            for layer in self.conformer_layers:
                x = layer(x, key_padding_mask=mask)
            return x.transpose(0, 1), lengths


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
    """Eğitim notebook'undaki ile birebir uyumlu — attribute adı 'ds' (NOT 'downsample')."""
    expansion = 1

    def __init__(self, ic: int, oc: int, stride: int = 1, ds: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(oc)
        self.conv2 = nn.Conv2d(oc, oc, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(oc)
        self.act = SiLU()
        self.ds = ds

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
        x = x.unsqueeze(1)
        x = self.frontend3D(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.avg_pool(self.trunk(x)).view(B, T, -1)
        x = self.proj(x)
        if input_lengths is None:
            input_lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
        x, _ = self.conformer(x, lengths=input_lengths)
        logits = self.ctc_fc(x).log_softmax(-1)
        return logits.permute(1, 0, 2)  # (T, B, V)
