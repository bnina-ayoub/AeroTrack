import tensorrt as trt
import torch
import torch.nn as nn
from aerotrack.models.network_blocks import EarlyExitDecisionModule

class DualTRTModel(nn.Module):
    # ADD early_exit_enabled TO INIT
    def __init__(self, early_trt, deep_trt, num_classes=1, img_hw=(640, 640), gate=None, early_exit_enabled=False):
        super().__init__()
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        self.rt = trt.Runtime(self.logger)
        
        with open(early_trt, "rb") as f:
            self.early_eng = self.rt.deserialize_cuda_engine(f.read())
        with open(deep_trt, "rb") as f:
            self.deep_eng = self.rt.deserialize_cuda_engine(f.read())
            
        self.early_ctx = self.early_eng.create_execution_context()
        self.deep_ctx = self.deep_eng.create_execution_context()
        
        self.gate = gate if gate is not None else EarlyExitDecisionModule()
        self.img_hw = img_hw
        self.n_ch = 5 + num_classes
        self.last_early_exit = False
        self.early_exit_enabled = early_exit_enabled  # <--- STORE THE FLAG
        self.is_trt_v3 = not hasattr(self.early_eng, "num_bindings")

    def forward(self, x):
        x = x.float().contiguous()
        batch, c, h, w = x.shape
        stream = torch.cuda.current_stream().cuda_stream
        
        early_anchors = (h // 8) * (w // 8)
        early_preds = torch.empty((batch, early_anchors, self.n_ch), dtype=torch.float32, device=x.device)
        dark3_feat = torch.empty((batch, 256, h // 8, w // 8), dtype=torch.float32, device=x.device)
        
        if self.is_trt_v3:
            self.early_ctx.set_tensor_address("image", x.data_ptr())
            self.early_ctx.set_tensor_address("early_preds", early_preds.data_ptr())
            self.early_ctx.set_tensor_address("dark3_feat", dark3_feat.data_ptr())
            self.early_ctx.execute_async_v3(stream_handle=stream)
        else:
            early_bindings = [int(0)] * self.early_eng.num_bindings
            for i in range(self.early_eng.num_bindings):
                name = self.early_eng.get_binding_name(i)
                if name == "image": early_bindings[i] = x.data_ptr()
                elif name == "early_preds": early_bindings[i] = early_preds.data_ptr()
                elif name == "dark3_feat": early_bindings[i] = dark3_feat.data_ptr()
            self.early_ctx.execute_async_v2(bindings=early_bindings, stream_handle=stream)
        
        torch.cuda.current_stream().synchronize()

        # FIX: ONLY RUN GATE IF EARLY EXIT IS ENABLED
        if self.early_exit_enabled and self.gate(early_preds, self.img_hw):
            self.last_early_exit = True
            return early_preds

        self.last_early_exit = False
            
        deep_anchors = early_anchors + (h // 16) * (w // 16) + (h // 32) * (w // 32)
        final_preds = torch.empty((batch, deep_anchors, self.n_ch), dtype=torch.float32, device=x.device)
        
        if self.is_trt_v3:
            self.deep_ctx.set_tensor_address("dark3_feat", dark3_feat.data_ptr())
            self.deep_ctx.set_tensor_address("final_preds", final_preds.data_ptr())
            self.deep_ctx.execute_async_v3(stream_handle=stream)
        else:
            deep_bindings = [int(0)] * self.deep_eng.num_bindings
            for i in range(self.deep_eng.num_bindings):
                name = self.deep_eng.get_binding_name(i)
                if name == "dark3_feat": deep_bindings[i] = dark3_feat.data_ptr()
                elif name == "final_preds": deep_bindings[i] = final_preds.data_ptr()
            self.deep_ctx.execute_async_v2(bindings=deep_bindings, stream_handle=stream)
        
        torch.cuda.current_stream().synchronize()
        return final_preds
