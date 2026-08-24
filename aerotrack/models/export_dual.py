import torch
import torch.nn as nn
from aerotrack.exp import get_exp
import torch.nn.functional as F
import math

class EarlyStage(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.stem = model.backbone.backbone.stem
        self.dark2 = model.backbone.backbone.dark2
        self.dark3 = model.backbone.backbone.dark3
        self.early_head = model.early_head

    def forward(self, x):
        stem_feat = self.stem(x)
        dark2_feat = self.dark2(stem_feat)
        dark3_feat = self.dark3(dark2_feat)
        early_preds = self.early_head([dark3_feat])
        return early_preds, dark3_feat

class DeepStage(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.dark4 = model.backbone.backbone.dark4
        self.dark5 = model.backbone.backbone.dark5
        self.pafpn = model.backbone
        self.head = model.head

    def forward(self, dark3_feat):
        dark4_feat = self.dark4(dark3_feat)
        dark5_feat = self.dark5(dark4_feat)
        pyramid = {"dark3": dark3_feat, "dark4": dark4_feat, "dark5": dark5_feat}
        fused = self.pafpn.forward_pafpn_only(pyramid)
        return self.head(fused)

# Load your trained PyTorch model
exp = get_exp("exps/aerotrack_proposed.py", None)
model = exp.get_model()
ckpt = torch.load("early_exit_weights.pth", map_location="cpu")
state_dict = ckpt["model_state_dict"]
model_state = model.state_dict()

for k in list(state_dict.keys()):
    if 'relative_position_bias_table' in k and k in model_state:
        ckpt_shape = state_dict[k].shape
        model_shape = model_state[k].shape

        if ckpt_shape != model_shape:
            num_heads = ckpt_shape[1]
            S_ckpt = int(math.sqrt(ckpt_shape[0]))
            S_model = int(math.sqrt(model_shape[0]))

            table = state_dict[k].view(S_ckpt, S_ckpt, num_heads).permute(2, 0, 1).unsqueeze(0)
            table = F.interpolate(table, size=(S_model, S_model), mode='bicubic', align_corners=False)
            state_dict[k] = table.squeeze(0).permute(1, 2, 0).view(-1, num_heads)

model.load_state_dict(state_dict, strict=False)
model.eval()

# 4. Dummy tensors for tracing
dummy_img = torch.randn(1, 3, 640, 640)
_, dummy_dark3 = EarlyStage(model)(dummy_img)

# 5. Export Early Stage
print("Exporting Early Stage to ONNX...")
torch.onnx.export(EarlyStage(model), dummy_img, "early_stage.onnx",
                  input_names=["image"], output_names=["early_preds", "dark3_feat"], opset_version=11)

# 6. Export Deep Stage
print("Exporting Deep Stage to ONNX...")
torch.onnx.export(DeepStage(model), dummy_dark3, "deep_stage.onnx",
                  input_names=["dark3_feat"], output_names=["final_preds"], opset_version=11)
print("Dual ONNX Export Complete!")
