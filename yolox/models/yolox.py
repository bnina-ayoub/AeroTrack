import torch
import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .network_blocks import EarlyExitDecisionModule

class YOLOX(nn.Module):
    def __init__(
        self,
        backbone=None,
        head=None,
        decision_gate_min_confidence=0.7,
        decision_gate_max_area=2025.0,
        decision_gate_uncertainty_lower=0.25,
        decision_gate_uncertainty_upper=0.30,
        empty_sky_threshold=0.25,
        early_exit_layer="dark3",
        early_exit_enabled=False
    ):
        super().__init__()
        self.backbone = backbone or YOLOPAFPN()
        self.head = head or YOLOXHead(80)
        self.early_exit_layer = early_exit_layer
        self.early_exit_enabled = early_exit_enabled
        self.last_early_exit = None

        layer_to_idx = {"dark3": 0, "dark4": 1, "dark5": 2}
        ch_idx = layer_to_idx[self.early_exit_layer]
        early_channels = self.head.in_channels[ch_idx]

        self.early_head = type(self.head)(
                num_classes=self.head.num_classes,
                width=self.head.width,
                in_channels=[early_channels],
                act=self.head.act
            )

        self.decision_gate = EarlyExitDecisionModule(
                min_confidence=decision_gate_min_confidence,
                max_area=decision_gate_max_area,
                uncertainty_lower=decision_gate_uncertainty_lower,
                uncertainty_upper=decision_gate_uncertainty_upper,
                empty_sky_threshold=empty_sky_threshold,
            )

    def forward(self, x, targets=None):
        if self.training:
            return self._forward_train(x, targets)
        return self._forward_inference(x)

    def _forward_inference(self, x):
        stem_features = self.backbone.backbone.stem(x)
        dark2_features = self.backbone.backbone.dark2(stem_features)
        dark3_features = self.backbone.backbone.dark3(dark2_features)

        dark4_features, dark5_features = None, None

        if self.early_exit_enabled:
            if self.early_exit_layer == "dark3":
                early_feat = dark3_features
            elif self.early_exit_layer == "dark4":
                dark4_features = self.backbone.backbone.dark4(dark3_features)
                early_feat = dark4_features
            elif self.early_exit_layer == "dark5":
                dark4_features = self.backbone.backbone.dark4(dark3_features)
                dark5_features = self.backbone.backbone.dark5(dark4_features)
                early_feat = dark5_features

            early_predictions = self.early_head([early_feat])
            img_hw = x.shape[-2], x.shape[-1]
            
            if self.decision_gate(early_predictions, img_hw):
                self.last_early_exit = True
                return early_predictions

            self.last_early_exit = False

        if dark4_features is None:
            dark4_features = self.backbone.backbone.dark4(dark3_features)
        if dark5_features is None:
            dark5_features = self.backbone.backbone.dark5(dark4_features)

        pyramid_features = {
            "dark3": dark3_features,
            "dark4": dark4_features,
            "dark5": dark5_features
        }

        fused_features = self.backbone.forward_pafpn_only(pyramid_features)
        return self.head(fused_features)

    def _forward_train(self, x, targets):
        stem_features = self.backbone.backbone.stem(x)
        dark2_features = self.backbone.backbone.dark2(stem_features)
        dark3_features = self.backbone.backbone.dark3(dark2_features)
        dark4_features = self.backbone.backbone.dark4(dark3_features)
        dark5_features = self.backbone.backbone.dark5(dark4_features)

        pyramid_features = {
            "dark3": dark3_features,
            "dark4": dark4_features,
            "dark5": dark5_features
        }

        fused_features = self.backbone.forward_pafpn_only(pyramid_features)

        main_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
            fused_features, targets, x
        )

        if self.early_exit_enabled:
            early_feat = pyramid_features[self.early_exit_layer]
            early_loss, _, _, _, _, _ = self.early_head(
                [early_feat], targets, x
            )
            total_loss = main_loss + early_loss
        else:
            early_loss = torch.tensor(0.0, device=main_loss.device)
            total_loss = main_loss

        return {
            "total_loss": total_loss,
            "main_loss": main_loss,
            "early_loss": early_loss,
            "iou_loss": iou_loss,
            "l1_loss": l1_loss,
            "conf_loss": conf_loss,
            "cls_loss": cls_loss,
            "num_fg": num_fg,
        }

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)