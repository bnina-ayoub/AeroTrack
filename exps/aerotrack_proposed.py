import os
from uavswarm_base import UAVSwarmBaseExperiment

class Exp(UAVSwarmBaseExperiment):
    def __init__(self):
        super().__init__()
        self.exp_name = "aerotrack"
        self.full_network_gflops = 36.14
        self.p3_only_gflops = 13.46

    def get_model(self):
        from aerotrack.models.yolo_pafpn import YOLOPAFPN
        from aerotrack.models.yolo_head import YOLOXHead
        from aerotrack.models.yolox import YOLOX
        
        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                depth=self.depth, 
                width=self.width, 
                in_channels=in_channels, 
                act="silu",
                enable_eca=True, 
                enable_str=True
            )
            head = YOLOXHead(
                num_classes=self.num_classes, 
                width=self.width, 
                in_channels=in_channels,
                act="silu"
            )
            
            self.model = YOLOX(
                backbone=backbone,
                head=head,
                decision_gate_min_confidence=0.75,
                decision_gate_max_area=1005.0,
                decision_gate_uncertainty_lower=0.25,
                decision_gate_uncertainty_upper=0.30,
                empty_sky_threshold=0.20,
                early_exit_layer="dark3",
                early_exit_enabled = getattr(self, "early_exit_enabled", False),
            )
        
        self.model.apply(self.configure_batch_normalization)
        self.model.head.initialize_biases(1e-2)
        
        return self.model