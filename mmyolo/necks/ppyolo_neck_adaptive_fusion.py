"""
Adaptive/Weighted feature fusion modules for PPYOLO in MMYOLO repository
- Implements ASFF (adaptive spatial feature fusion) and MFWF (multi-scale feature weighted fusion)
- Provides helper `attach_fusion_to_ppyolo_neck` showing how to integrate into PPYOLOECSPPAFPN

How to use (summary):
1. Put this file under mmyolo/models/necks/ (or project equivalent).
2. Import and in PPYOLOECSPPAFPN __init__, add a `fusion_mode` argument and attach one of these modules:
       self.fusion = ASFF(channels_list, target_level=..., reduction=8)  # or MFWF(...)
   Then in forward, instead of `torch.cat(...)` use `self.fusion([feat_small, feat_mid, feat_large])`

Notes:
- This code aims to be drop-in friendly but you'll need to adapt exact channel orders and up/downsample helpers to your
  project's conventions (some PPYOLO variants store P3,P4,P5 etc.).
- Tests: run a quick forward pass with dummy tensors to ensure shapes match.

References:
- ASFF: https://arxiv.org/abs/1911.09516
- MFWF concept: learn per-scale weights (simpler implementation here)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ASFFLevel(nn.Module):
    """ASFF module for one target level. Accepts list of features [p3, p4, p5] (or similar)
    and fuses them adaptively.

    This is a pragmatic implementation (not an exact reproduction) designed to be easy to plug in.
    """

    def __init__(self, in_channels_list, out_channels, target_level=1, reduction=8):
        """
        in_channels_list: list of ints, channels for each scale (ordered from highest res to lowest res)
        out_channels: channels for fused output
        target_level: index of the target level to output (0..N-1). ASFF creates weights for all inputs
        reduction: internal reduction for weight computation
        """
        super().__init__()
        assert len(in_channels_list) >= 2, "need at least two scales"
        self.num_scales = len(in_channels_list)
        self.target_level = target_level

        # Align convs: transform each input to the target spatial size and to a common channel dimension
        self.resizers = nn.ModuleList()
        for in_ch in in_channels_list:
            # use 1x1 conv to reduce channels and then resize in forward
            self.resizers.append(ConvBNAct(in_ch, out_channels, k=1, s=1, p=0))

        # weight generation: for each scale produce a single-channel weight map
        # We'll compute shared feature then per-scale conv to weight
        self.weight_compress = ConvBNAct(out_channels * self.num_scales, out_channels // reduction, k=1, s=1, p=0)
        self.weight_predictors = nn.ModuleList([
            nn.Conv2d(out_channels // reduction, 1, kernel_size=1)
            for _ in range(self.num_scales)
        ])

        # final conv to mix
        self.output_conv = ConvBNAct(out_channels, out_channels, k=3, s=1, p=1)

    def forward(self, feats):
        # feats: list of tensors [f0 (high res), f1, f2 (low res)]
        # target spatial size is that of feats[target_level]
        target_size = feats[self.target_level].shape[-2:]

        resized = []
        for i, f in enumerate(feats):
            x = self.resizers[i](f)  # channel align
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            resized.append(x)

        # channel-wise concat for weight computation
        cat = torch.cat(resized, dim=1)
        shared = self.weight_compress(cat)

        weight_maps = [pred(shared) for pred in self.weight_predictors]  # list of [B,1,H,W]
        weight_stack = torch.cat(weight_maps, dim=1)  # [B, S, H, W]
        weight_soft = F.softmax(weight_stack, dim=1)  # softmax across scales

        # weighted sum
        fused = 0
        for i in range(self.num_scales):
            w = weight_soft[:, i:i+1, ...]
            fused = fused + resized[i] * w

        out = self.output_conv(fused)
        return out


class ASFF(nn.Module):
    """ASFF wrapper for multiple target levels.
    If you only need fusion to a single level (e.g., keep highest-level context), use a single ASFFLevel.
    """

    def __init__(self, in_channels_list, out_channels, reduction=8):
        super().__init__()
        self.levels = nn.ModuleList()
        num = len(in_channels_list)
        for target in range(num):
            self.levels.append(ASFFLevel(in_channels_list, out_channels, target_level=target, reduction=reduction))

    def forward(self, feats):
        # returns list of fused features at each level, in same order as input
        outs = []
        for lv in self.levels:
            outs.append(lv(feats))
        return outs


class MFWF(nn.Module):
    """Multi-scale Feature Weighted Fusion (simpler, channel-wise learned weights)

    Strategy:
      - For each input scale, compute a per-channel scale weight using global pooling and small FC
      - Normalize weights across scales (softmax) and produce weighted sum (after resizing spatially)
    """

    def __init__(self, in_channels_list, out_channels=None, reduction=16):
        super().__init__()
        self.num_scales = len(in_channels_list)
        # Use out_channels if specified else keep channels of target (assume first)
        if out_channels is None:
            out_channels = in_channels_list[0]
        self.out_channels = out_channels

        # Align channels
        self.align_convs = nn.ModuleList([ConvBNAct(ch, out_channels, k=1, s=1, p=0) for ch in in_channels_list])

        # per-scale channel weight generators
        self.fc_generators = nn.ModuleList()
        for _ in in_channels_list:
            self.fc_generators.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=False)
                )
            )

        self.final_conv = ConvBNAct(out_channels, out_channels, k=3, s=1, p=1)

    def forward(self, feats):
        # feats: list of tensors
        target_size = feats[0].shape[-2:]  # choose highest-res as target by default

        aligned = []
        for i, f in enumerate(feats):
            x = self.align_convs[i](f)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            aligned.append(x)

        # compute channel logits per scale
        logits = []
        for i, x in enumerate(aligned):
            l = self.fc_generators[i](x)  # [B, C, 1, 1]
            logits.append(l)

        logits_stack = torch.stack(logits, dim=1)  # [B, S, C, 1, 1]
        # normalize across scales for each channel -> softmax on dim=1
        logits_stack = logits_stack.squeeze(-1).squeeze(-1)  # [B, S, C]
        weight = F.softmax(logits_stack, dim=1)  # [B, S, C]

        # apply weights and sum
        fused = 0
        for i in range(self.num_scales):
            w = weight[:, i:i+1, :].unsqueeze(-1).unsqueeze(-1)  # [B,1,C,1,1]
            fused = fused + aligned[i] * w

        # fused shape [B, C, H, W]
        out = self.final_conv(fused)
        return out


# Integration helper

def attach_fusion_to_ppyolo_neck(neck_module, mode='asff', out_channels=None, reduction=8):
    """
    neck_module: instance of PPYOLOECSPPAFPN (or similar) -- not the class
    mode: 'asff' or 'mfwf'
    out_channels: desired output channel size for fused features. If None, inferred from neck

    This function will attach `neck_module.fusion` and provide guidance for modifying forward.

    Example modification in PPYOLOECSPPAFPN.__init__:
        self.fusion_mode = fusion_mode  # e.g. 'asff'
        self.fusion = None
        if fusion_mode is not None:
            from .ppyolo_neck_adaptive_fusion import attach_fusion_to_ppyolo_neck
            attach_fusion_to_ppyolo_neck(self, mode=fusion_mode, out_channels=some_ch)

    And in forward, replace concatenation logic (where features are fused) with:
        if self.fusion is not None:
            fused = self.fusion([feat_high_res, feat_mid, feat_low])  # depending on ordering
            # if ASFF returns list (for each level), adapt accordingly

    NOTE: Adjust the order of features passed according to your neck's indexing.
    """
    # try to detect channel sizes from neck attributes (common names)
    chs = []
    candidates = [
        getattr(neck_module, 'in_channels', None),
        getattr(neck_module, 'out_channels', None),
    ]
    # fallback: try to inspect some feature extraction conv layers
    # This detection is best-effort; you may want to pass channels explicitly.
    if out_channels is None:
        # attempt to infer channels from neck convs if available
        for name, m in neck_module.named_modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
                chs.append(m.out_channels)
                if len(chs) >= 3:
                    break
        if chs:
            out_channels = chs[0]
        else:
            raise RuntimeError('Could not infer out_channels for fusion; please pass out_channels explicitly')

    # For typical PPYOLO neck we fuse three scales: [P3, P4, P5]
    # Expect channel sizes are equal; if not, user should pass explicit in_channels_list
    # Provide a conservative default: use neck's out_channels if exists
    if hasattr(neck_module, 'out_channels') and neck_module.out_channels is not None:
        default_ch = neck_module.out_channels
    else:
        default_ch = out_channels

    # create in_channels_list for 3 scales
    in_channels_list = [default_ch] * 3

    if mode.lower() == 'asff':
        neck_module.fusion = ASFF(in_channels_list, out_channels=out_channels, reduction=reduction)
    elif mode.lower() == 'mfwf' or mode.lower() == 'mfwf':
        neck_module.fusion = MFWF(in_channels_list, out_channels=out_channels, reduction=reduction)
    else:
        raise ValueError('Unknown fusion mode: ' + str(mode))

    # attach meta info so forward patch is easier
    neck_module._fusion_mode = mode.lower()
    neck_module._fusion_channels = out_channels

    return neck_module


# Quick sanity test (run standalone)
if __name__ == '__main__':
    # create dummy features: P3 (80x80), P4 (40x40), P5 (20x20) for input image 640
    b = 2
    C = 256
    p3 = torch.rand(b, C, 80, 80)
    p4 = torch.rand(b, C, 40, 40)
    p5 = torch.rand(b, C, 20, 20)

    asff = ASFF([C, C, C], out_channels=C)
    outs = asff([p3, p4, p5])
    print('ASFF outputs:', [o.shape for o in outs])

    mf = MFWF([C, C, C], out_channels=C)
    out_m = mf([p3, p4, p5])
    print('MFWF output:', out_m.shape)
