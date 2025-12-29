# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig
from mmyolo.models.backbones import BaseBackbone
from mmyolo.models.layers.yolo_bricks import CSPResLayer
from mmyolo.registry import MODELS


class SELayer(nn.Module):
    """Lightweight Squeeze-and-Excitation (SE) block for shallow feature boosting."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)
        s = self.sigmoid(s)
        return x * s


@MODELS.register_module()
class PPYOLOECSPResNetSEStage1(BaseBackbone):
    """PPYOLOE-CSPResNet backbone with SE attention inserted at Stage 1."""

    arch_settings = {
        'P5': [[64, 128, 3], [128, 256, 6], [256, 512, 6], [512, 1024, 3]],
    }

    def __init__(self,
                 arch: str = 'P5',
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 plugins: Union[dict, List[dict]] = None,
                 arch_ovewrite: dict = None,
                 block_cfg: ConfigType = dict(
                     type='PPYOLOEBasicBlock', shortcut=True, use_alpha=True),
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 attention_cfg: ConfigType = dict(
                     type='EffectiveSELayer', act_cfg=dict(type='HSigmoid')),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None,
                 use_large_stem: bool = False,
                 se_reduction: int = 16):
        # Build base architecture settings
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        arch_setting = [[
            int(in_channels * widen_factor),
            int(out_channels * widen_factor),
            round(num_blocks * deepen_factor)
        ] for in_channels, out_channels, num_blocks in arch_setting]

        self.block_cfg = block_cfg
        self.use_large_stem = use_large_stem
        self.attention_cfg = attention_cfg
        self.se_reduction = se_reduction

        super().__init__(
            arch_setting,
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

        # Insert SE after Stage 1 output
        out_channels_stage1 = arch_setting[0][1]
        self.se_stage1 = SELayer(out_channels_stage1, reduction=se_reduction)

    def build_stem_layer(self):
        if self.use_large_stem:
            stem = nn.Sequential(
                ConvModule(
                    self.input_channels,
                    self.arch_setting[0][0] // 2,
                    3,
                    stride=2,
                    padding=1,
                    act_cfg=self.act_cfg,
                    norm_cfg=self.norm_cfg),
                ConvModule(
                    self.arch_setting[0][0] // 2,
                    self.arch_setting[0][0] // 2,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    self.arch_setting[0][0] // 2,
                    self.arch_setting[0][0],
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        else:
            stem = nn.Sequential(
                ConvModule(
                    self.input_channels,
                    self.arch_setting[0][0] // 2,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    self.arch_setting[0][0] // 2,
                    self.arch_setting[0][0],
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        return stem

    def build_stage_layer(self, stage_idx: int, setting: list):
        in_channels, out_channels, num_blocks = setting
        cspres_layer = CSPResLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            num_block=num_blocks,
            block_cfg=self.block_cfg,
            stride=2,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            attention_cfg=self.attention_cfg,
            use_spp=False)
        return [cspres_layer]

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            # Apply SE only after Stage 1 (index 1 in self.layers)
            if layer_name == 'stage1':
                x = self.se_stage1(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
