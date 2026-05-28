# @Time    : 2026/05/26  10:00
# @Author  : Kyrietop11
# @File    : bdsfusion.py
# @Software: VScode

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import torch
import torch.nn as nn

from .CSAMamba import CSAMamba
from .SFCIM import SFCIM
from .modules import PatchEmbed, PatchReconstruct


@dataclass
class BDSFusionConfig:
    in_channels: int = 1
    out_channels: int = 1
    embed_dim: int = 36
    patch_size: int = 2
    depths: Sequence[int] = field(default_factory=lambda: (3,))
    recon_count: int = 2
    sfcim_iters: int = 4
    heads: int = 4
    drop_path: float = 0.0
    dropout: float = 0.0
    window_size: int = 7
    # CSAMamba SS2D parameters
    d_state: int = 24
    num_ldconv: int = 2
    ldconv_num_param: int = 3
    # SFCIM parameters
    sfcim_d_state: int = 24


class BDSFusion(nn.Module):
    """Bidirectionally driven saliency fusion network.

    Three-stage design: parallel CSA-CM feature extraction,
    iterative SFCIM feature integration, and CSA-CM reconstruction.
    """

    def __init__(self, config: BDSFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or BDSFusionConfig()
        cfg = self.config
        self.mwir_embed = PatchEmbed(cfg.in_channels, cfg.embed_dim, cfg.patch_size)
        self.lwir_embed = PatchEmbed(cfg.in_channels, cfg.embed_dim, cfg.patch_size)

        depth_count = sum(int(d) for d in cfg.depths)
        csa_kwargs = dict(
            head_num=cfg.heads, drop_path=cfg.drop_path, attn_drop_rate=cfg.dropout,
            window_size=cfg.window_size, d_state=cfg.d_state,
            num_ldconv=cfg.num_ldconv, ldconv_num_param=cfg.ldconv_num_param,
        )
        self.mwir_extraction = nn.ModuleList(
            [CSAMamba(cfg.embed_dim, **csa_kwargs) for _ in range(depth_count)]
        )
        self.lwir_extraction = nn.ModuleList(
            [CSAMamba(cfg.embed_dim, **csa_kwargs) for _ in range(depth_count)]
        )
        self.feature_integration = nn.ModuleList(
            [SFCIM(hidden_dim=cfg.embed_dim, drop_path=cfg.drop_path,
                   attn_drop_rate=cfg.dropout, d_state=cfg.sfcim_d_state)
             for _ in range(cfg.sfcim_iters)]
        )
        self.reconstruction_refine = nn.ModuleList(
            [CSAMamba(cfg.embed_dim, **csa_kwargs) for _ in range(cfg.recon_count)]
        )
        self.inverse_patch_embedding = PatchReconstruct(cfg.embed_dim, cfg.out_channels, cfg.patch_size)

    def forward_features(self, mwir: torch.Tensor, lwir: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if mwir.shape != lwir.shape:
            raise ValueError(f"MWIR and LWIR tensors must have identical shapes, got {mwir.shape} and {lwir.shape}.")
        m_tokens, spatial_size = self.mwir_embed(mwir)
        l_tokens, l_spatial = self.lwir_embed(lwir)
        if spatial_size != l_spatial:
            raise ValueError("Patch embeddings produced mismatched spatial sizes.")

        for block in self.mwir_extraction:
            m_tokens = block(m_tokens, spatial_size=spatial_size)
        for block in self.lwir_extraction:
            l_tokens = block(l_tokens, spatial_size=spatial_size)

        fused = m_tokens
        for block in self.feature_integration:
            m_tokens, l_tokens, fused = block(m_tokens, l_tokens, spatial_size=spatial_size)
        for block in self.reconstruction_refine:
            fused = block(fused, spatial_size=spatial_size)
        return fused, spatial_size

    def forward(self, mwir: torch.Tensor, lwir: torch.Tensor) -> torch.Tensor:
        fused_tokens, spatial_size = self.forward_features(mwir, lwir)
        return self.inverse_patch_embedding(fused_tokens, spatial_size, output_size=mwir.shape[-2:])
