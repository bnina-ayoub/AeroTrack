import os
from exps.uavswarm_base import UAVSwarmBaseExperiment

class Exp(UAVSwarmBaseExperiment):
    def __init__(self):
        super().__init__()
        self.exp_name = "baseline_eca_swin"
        self.tracking_distance_metric = "nwd"

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
                early_exit_enabled=False
            )
        
        self.model.apply(self.configure_batch_normalization)
        self.model.head.initialize_biases(1e-2)
        
        return self.model