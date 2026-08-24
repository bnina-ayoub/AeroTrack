import tensorrt as trt

logger = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_path, trt_path):
    builder = trt.Builder(logger)
    
    # 1. Cross-Compatible Network Creation (Explicit Batch for TRT 8)
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
    else:
        network = builder.create_network() # TRT 10+
        
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    # 2. Cross-Compatible Workspace Memory Allocation
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    else:
        config.max_workspace_size = 4 << 30

    # 3. Cross-Compatible FP16 Enabling
    if builder.platform_has_fast_fp16:
        if hasattr(trt.BuilderFlag, "FP16"):
            config.set_flag(trt.BuilderFlag.FP16)
        elif hasattr(trt.BuilderFlag, "kFP16"):
            config.set_flag(trt.BuilderFlag.kFP16)

    print(f"Parsing {onnx_path}...")
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError(f"Failed to parse {onnx_path}")

    print(f"Building TensorRT engine {trt_path} (this takes a minute or two)...")
    
    # 4. Cross-Compatible Engine Serialization
    if hasattr(builder, "build_serialized_network"):
        engine_bytes = builder.build_serialized_network(network, config)
        with open(trt_path, "wb") as f:
            f.write(engine_bytes)
    else:
        engine = builder.build_engine(network, config)
        with open(trt_path, "wb") as f:
            f.write(engine.serialize())

    print(f"Successfully built {trt_path}!\n")

# Compile both engines
build_engine("early_stage.onnx", "early_stage_fp16.trt")
build_engine("deep_stage.onnx", "deep_stage_fp16.trt")
