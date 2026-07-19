#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock as TimmSwinTransformerBlock

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu", enable_eca=False
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class EfficientChannelAttention(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # [B,C,H,W] -> [B,C,1,1]
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                          # [B,C,1,1]
        y = y.squeeze(-1).transpose(-1, -2)          # [B,1,C]
        y = self.conv(y)                              # [B,1,C]
        y = y.transpose(-1, -2).unsqueeze(-1)        # [B,C,1,1]
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
        enable_eca=False
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
    
        self.eca = EfficientChannelAttention(hidden_channels) if enable_eca else None
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv1(x)
        if self.eca is not None:
            y = self.eca(y)
        y = self.conv2(y)
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
        enable_eca=False
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act, enable_eca=enable_eca
            ) if enable_eca else Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class CSPSTRLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        input_resolution,
        expansion=0.5,
        depth=2,
        num_heads=8,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.str = SwinTokenStage(hidden_channels, input_resolution=input_resolution, depth=depth, num_heads=num_heads)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.str(x_1)
        x_3 = self.conv2(x)
        x = torch.cat((x_2, x_3), dim=1)
        return self.conv3(x)



class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)

 
class SwinTokenStage(nn.Module):
    """
    YOLOX feature-map wrapper around timm SwinTransformerBlock.
    Input/Output: [B, C, H, W]
    """

    def __init__(
        self,
        dim,
        input_resolution,   # tuple(H, W) for this stage
        depth=2,
        num_heads=8,
        window_size=7,
        mlp_ratio=4.0,
        drop_path=0.0,
    ):
        super().__init__()
        H, W = input_resolution
        blocks = []
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            blocks.append(
                TimmSwinTransformerBlock(
                    dim=dim,
                    input_resolution=(H, W),
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                    qkv_bias=True,
                    dynamic_mask=True,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.blocks(x)
        # back to [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    

class EarlyExitDecisionModule(nn.Module):
    def __init__(
        self,
        min_confidence=0.65,
        max_area=2025.0,
        uncertainty_lower=0.25,
        uncertainty_upper=0.60,
        empty_sky_threshold=0.1,
    ):
        super().__init__()
        self.min_confidence = min_confidence
        self.max_area = max_area
        self.uncertainty_lower = uncertainty_lower
        self.uncertainty_upper = uncertainty_upper
        # new: if the maximum objectness is below this, treat as empty sky -> early exit
        self.empty_sky_threshold = empty_sky_threshold

    def forward(self, predictions: torch.Tensor, img_hw: tuple) -> bool:
        # predictions: [B, N, 5+cls], use per-batch or whole-batch stats as before
        H, W = img_hw
        objectness_scores = predictions[..., 4]
        widths = predictions[..., 2]
        heights = predictions[..., 3]

        box_areas = widths * heights

        # 1) Empty-sky fast path: if even the maximum objectness is very low,
        #    we can confidently early-exit and skip the expensive neck/head.
        if objectness_scores.max() < self.empty_sky_threshold:
            return True

        # 2) If there are no confident detections (above min_confidence),
        #    treat the frame as ambiguous and do the full network.
        confident_mask = objectness_scores > self.min_confidence
        if not confident_mask.any():
            return False

        # 3) If there are confident detections but some are unreasonably large,
        #    avoid early exit (likely false positives / big objects need full head).
        if (box_areas[confident_mask] > self.max_area).any():
            return False

        # 4) Uncertainty check: if any anchor falls into the uncertainty band,
        #    avoid early exit and run full network.
        has_high_uncertainty_regions = (
            ((objectness_scores > self.uncertainty_lower) & (objectness_scores < self.uncertainty_upper))
        ).any()
        if has_high_uncertainty_regions:
            return False

        # All checks passed -> early exit
        return True