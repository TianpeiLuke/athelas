---
tags:
  - design
  - performance
  - onnx_runtime
  - inference_optimization
  - pytorch_models
keywords:
  - ONNX Runtime optimization
  - inference latency reduction
  - BERT model optimization
  - trimodal BERT performance
  - SessionOptions configuration
  - graph optimization
  - quantization
  - TensorRT integration
topics:
  - ONNX Runtime performance
  - model inference optimization
  - PyTorch to ONNX conversion
  - production ML serving
language: python
date of note: 2025-12-07
---

# ONNX Runtime Optimization for Trimodal BERT Models

## Overview

This document proposes a comprehensive optimization strategy for ONNX Runtime to reduce inference latency of trimodal BERT models from **335-545ms to 50-100ms per request** (4-6x speedup). The current implementation uses a basic ONNX Runtime configuration with no optimizations, resulting in suboptimal performance for production serving.

## Problem Statement

### Current Performance Issues

Based on production logs and code analysis, the current ONNX inference implementation exhibits:

1. **High Latency**: 335-545ms per single inference request
2. **No Graph Optimizations**: ONNX graph not optimized for target hardware
3. **Unoptimized Threading**: Default thread configuration not tuned for workload
4. **No Hardware Acceleration**: Missing TensorRT/CUDA optimizations
5. **FP32 Precision**: No quantization applied (4x larger models, slower inference)
6. **Dual BERT Encoders**: Sequential processing doubles inference time

### Current Implementation Analysis

From `projects/rnr_pytorch_bedrock/dockers/lightning_models/utils/pl_train.py`:

```python
def load_onnx_model(onnx_path: Union[str, Path]) -> ort.InferenceSession:
    """Load ONNX model with MINIMAL configuration - NO OPTIMIZATIONS!"""
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )
    
    try:
        # ❌ CRITICAL ISSUE: No SessionOptions configured!
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        logger.info(f"Successfully loaded ONNX model from {onnx_path}")
        return session
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")
```

**Key Problems:**
- No `SessionOptions` configured
- No graph optimization level specified
- No thread tuning for CPU/GPU workloads
- No memory optimizations enabled
- Missing execution mode configuration
- No model-specific optimizations (BERT fusion)

### Architecture Bottlenecks

From `projects/rnr_pytorch_bedrock/dockers/lightning_models/trimodal/pl_trimodal_bert.py`:

**Computational Overhead Sources:**
1. **Dual BERT Encoders**: Primary (dialogue) + Secondary (shiptrack) text encoders
2. **TabAE Network**: Tabular feature processing
3. **Fusion Network**: Multiple linear layers for tri-modal fusion
4. **Softmax Layer**: Final probability computation

**Inference Pipeline:**
```
Input → Primary BERT (150-200ms) → Secondary BERT (150-200ms) → TabAE (10ms) → Fusion (10ms) → Softmax (5ms)
Total: ~335-545ms per request
```

## Root Cause Analysis

### 1. Missing ONNX Runtime Optimizations

**Impact: 2-3x slowdown**

The current implementation uses default ONNX Runtime settings with no performance tuning:

```python
# Current implementation - NO optimizations
session = ort.InferenceSession(str(onnx_path), providers=providers)

# Missing critical configurations:
# - Graph optimization level (ORT_ENABLE_ALL)
# - Execution mode (ORT_SEQUENTIAL vs ORT_PARALLEL)
# - Thread pool configuration
# - Memory pattern optimization
# - CPU memory arena allocation
```

### 2. Unoptimized BERT Operations

**Impact: 2-4x slowdown for BERT layers**

ONNX Runtime has specialized BERT optimizations (layer fusion, attention optimization) that are not applied:

```python
# Missing BERT-specific optimizations:
# - Multi-head attention fusion
# - Layer normalization fusion
# - GELU activation fusion
# - Embedding layer optimization
# - Skip connection fusion
```

### 3. No Quantization Applied

**Impact: 2-4x slowdown + 4x memory overhead**

Models run in FP32 precision without INT8 quantization:
- **FP32 Model**: 4 bytes per parameter, slower computation
- **INT8 Model**: 1 byte per parameter, 2-4x faster inference
- **Memory**: 4x reduction with INT8 quantization

### 4. Suboptimal Hardware Utilization

**Impact: 1.5-2x slowdown**

Missing hardware-specific acceleration:
- No TensorRT execution provider (GPU optimization)
- No thread affinity tuning
- No SIMD instruction utilization
- Unoptimized batch processing

## Proposed Solution

### Phase 1: ONNX Runtime SessionOptions Optimization (HIGH IMPACT)

**Expected Speedup: 2-3x (335ms → 110-170ms)**

Implement comprehensive SessionOptions configuration:

```python
def load_optimized_onnx_model(
    onnx_path: Union[str, Path],
    enable_profiling: bool = False,
    inter_op_threads: int = 1,
    intra_op_threads: int = 4,
) -> ort.InferenceSession:
    """
    Load ONNX model with production-grade optimization settings.
    
    Args:
        onnx_path: Path to ONNX model file
        enable_profiling: Enable performance profiling for debugging
        inter_op_threads: Number of threads for parallel operator execution
        intra_op_threads: Number of threads within operators (e.g., matrix operations)
    
    Returns:
        Optimized ONNX Runtime InferenceSession
    """
    # ===== 1. Configure SessionOptions =====
    sess_options = ort.SessionOptions()
    
    # Graph optimization: Enable all optimizations (level 99)
    # - Constant folding
    # - Redundant node elimination  
    # - Operator fusion (BERT-specific)
    # - Shape inference
    # - Common subexpression elimination
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Execution mode: Sequential for single-request serving
    # ORT_SEQUENTIAL: Operators run sequentially (lower latency, single request)
    # ORT_PARALLEL: Operators run in parallel (higher throughput, batch requests)
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    # Memory optimizations
    sess_options.enable_mem_pattern = True  # Reuse memory allocations
    sess_options.enable_cpu_mem_arena = True  # Use memory arena for faster allocation
    
    # Thread configuration (tune based on CPU cores)
    sess_options.intra_op_num_threads = intra_op_threads  # Threads per operator
    sess_options.inter_op_num_threads = inter_op_threads  # Threads for parallel ops
    
    # Profiling (disable in production for performance)
    if enable_profiling:
        sess_options.enable_profiling = True
        sess_options.profile_file_prefix = "onnx_profile"
    
    # ===== 2. Configure Execution Providers =====
    providers = []
    
    # Try TensorRT first (best GPU performance)
    if ort.get_device() == "GPU":
        try:
            providers.append(
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": True,  # Enable FP16 precision (2x faster)
                        "trt_engine_cache_enable": True,  # Cache compiled engines
                        "trt_engine_cache_path": "/tmp/trt_cache",
                        "trt_max_workspace_size": 2147483648,  # 2GB workspace
                    },
                )
            )
        except Exception as e:
            logger.warning(f"TensorRT provider not available: {e}")
    
    # Fallback to CUDA
    if ort.get_device() == "GPU":
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB limit
                    "cudnn_conv_algo_search": "EXHAUSTIVE",  # Find best convolution algorithm
                    "do_copy_in_default_stream": True,
                },
            )
        )
    
    # CPU fallback
    providers.append("CPUExecutionProvider")
    
    # ===== 3. Create Optimized Session =====
    try:
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=providers,
        )
        
        logger.info(f"✓ Loaded ONNX model with optimizations from {onnx_path}")
        logger.info(f"  Graph optimization: ORT_ENABLE_ALL")
        logger.info(f"  Execution mode: ORT_SEQUENTIAL")
        logger.info(f"  Intra-op threads: {intra_op_threads}")
        logger.info(f"  Inter-op threads: {inter_op_threads}")
        logger.info(f"  Providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")
        logger.info(f"  Expected inputs: {[inp.name for inp in session.get_inputs()]}")
        
        return session
        
    except Exception as e:
        raise RuntimeError(f"Failed to load optimized ONNX model: {e}")
```

**Key Optimizations:**

1. **Graph Optimization Level**: `ORT_ENABLE_ALL` enables all graph transformations
   - Constant folding: Pre-compute constant operations
   - Operator fusion: Combine multiple ops into single kernel
   - Dead code elimination: Remove unused operations
   - Shape inference: Optimize tensor shapes

2. **Execution Mode**: `ORT_SEQUENTIAL` for low-latency single-request serving
   - Sequential execution: Operators run one at a time
   - Lower latency: Reduced scheduling overhead
   - Better for real-time inference

3. **Memory Optimizations**:
   - `enable_mem_pattern`: Reuse memory buffers across inferences
   - `enable_cpu_mem_arena`: Fast memory allocation from pre-allocated pool

4. **Thread Configuration**:
   - `intra_op_num_threads=4`: Parallel execution within operators (matrix ops)
   - `inter_op_num_threads=1`: Sequential operator execution (lower latency)

### Phase 2: BERT Model Fusion Optimization (HIGH IMPACT)

**Expected Speedup: 1.5-2x additional (110ms → 55-75ms)**

Apply BERT-specific graph optimizations using ONNX Runtime transformer tools:

```python
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions

def optimize_bert_model(
    input_model_path: str,
    output_model_path: str,
    model_type: str = "bert",
    num_heads: int = 12,
    hidden_size: int = 768,
) -> None:
    """
    Apply BERT-specific optimizations to ONNX model.
    
    This creates a new optimized model file - original model is preserved.
    Optimizations are mathematically equivalent (output identical within FP precision).
    
    Args:
        input_model_path: Path to original ONNX model
        output_model_path: Path to save optimized model
        model_type: Model architecture type ("bert", "gpt2", "bart")
        num_heads: Number of attention heads in BERT model
        hidden_size: Hidden dimension size in BERT model
    """
    logger.info(f"Optimizing BERT model: {input_model_path}")
    
    # Configure fusion options
    fusion_options = FusionOptions(model_type)
    fusion_options.enable_gelu = True  # Fuse GELU activation
    fusion_options.enable_layer_norm = True  # Fuse layer normalization
    fusion_options.enable_attention = True  # Fuse multi-head attention
    fusion_options.enable_skip_layer_norm = True  # Fuse skip connections + layer norm
    fusion_options.enable_embed_layer_norm = True  # Fuse embedding + layer norm
    fusion_options.enable_bias_skip_layer_norm = True  # Fuse bias + skip + layer norm
    fusion_options.enable_bias_gelu = True  # Fuse bias + GELU
    fusion_options.enable_gelu_approximation = False  # Use exact GELU (better accuracy)
    
    # Create optimizer
    optimizer_instance = optimizer.optimize_model(
        input=input_model_path,
        model_type=model_type,
        num_heads=num_heads,
        hidden_size=hidden_size,
        optimization_options=fusion_options,
        opt_level=99,  # Maximum optimization level
        use_gpu=torch.cuda.is_available(),
    )
    
    # Save optimized model
    optimizer_instance.save_model_to_file(output_model_path)
    
    logger.info(f"✓ Saved optimized BERT model to {output_model_path}")
    logger.info(f"  Original nodes: {len(optimizer_instance.model.graph.node)}")
    logger.info(f"  Fused attention blocks: {optimizer_instance.get_fused_operator_statistics()}")
    
    return optimizer_instance


def load_bert_optimized_model(model_dir: str) -> ort.InferenceSession:
    """
    Load BERT-optimized ONNX model with SessionOptions.
    
    Workflow:
    1. Check for pre-optimized model (model_optimized.onnx)
    2. If not found, optimize original model (model.onnx)
    3. Load with optimized SessionOptions
    
    Args:
        model_dir: Directory containing ONNX models
    
    Returns:
        Optimized ONNX Runtime session
    """
    original_model = Path(model_dir) / "model.onnx"
    optimized_model = Path(model_dir) / "model_optimized.onnx"
    
    # Create optimized model if it doesn't exist
    if not optimized_model.exists():
        logger.info("Optimized model not found, creating from original...")
        optimize_bert_model(
            input_model_path=str(original_model),
            output_model_path=str(optimized_model),
            model_type="bert",
            num_heads=12,  # Standard BERT-base configuration
            hidden_size=768,
        )
    
    # Load optimized model with SessionOptions
    return load_optimized_onnx_model(optimized_model)
```

**BERT Fusion Transformations:**

Before optimization:
```
LayerNorm → Dropout → Add → GELU → Linear → LayerNorm → Attention → ...
(10+ separate operations)
```

After optimization:
```
[FusedBertLayer] → [FusedMultiHeadAttention] → ...
(2-3 fused operations)
```

**Benefits:**
- **Reduced Memory Transfers**: Fused operations avoid intermediate tensor materialization
- **Improved Cache Locality**: Fused kernels better utilize L1/L2 cache
- **Lower Scheduling Overhead**: Fewer kernel launches (GPU) or function calls (CPU)
- **SIMD Optimization**: Fused operations can use wider vector instructions

### Phase 3: INT8 Quantization (MEDIUM IMPACT)

**Expected Speedup: 1.5-2x additional (55ms → 28-40ms)**

Apply dynamic quantization for 2-4x speedup with minimal accuracy loss:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(
    input_model_path: str,
    output_model_path: str,
    per_channel: bool = True,
    reduce_range: bool = False,
) -> None:
    """
    Apply INT8 quantization to ONNX model.
    
    Dynamic quantization converts weights to INT8 and performs runtime
    activation quantization. Results in 4x smaller models and 2-4x faster inference.
    
    IMPORTANT: Validate accuracy after quantization!
    
    Args:
        input_model_path: Path to original FP32 model
        output_model_path: Path to save quantized INT8 model
        per_channel: Use per-channel quantization (better accuracy)
        reduce_range: Reduce quantization range for better accuracy (slightly slower)
    """
    logger.info(f"Quantizing model: {input_model_path}")
    
    try:
        quantize_dynamic(
            model_input=input_model_path,
            model_output=output_model_path,
            weight_type=QuantType.QInt8,  # Quantize weights to INT8
            per_channel=per_channel,
            reduce_range=reduce_range,
            extra_options={
                'EnableSubgraph': True,  # Quantize subgraphs
                'WeightSymmetric': True,  # Symmetric quantization
            }
        )
        
        # Verify quantized model
        import onnx
        quantized_model = onnx.load(output_model_path)
        onnx.checker.check_model(quantized_model)
        
        # Compare model sizes
        original_size = Path(input_model_path).stat().st_size / (1024 * 1024)
        quantized_size = Path(output_model_path).stat().st_size / (1024 * 1024)
        
        logger.info(f"✓ Quantization completed successfully")
        logger.info(f"  Original model: {original_size:.2f} MB")
        logger.info(f"  Quantized model: {quantized_size:.2f} MB")
        logger.info(f"  Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        raise


def validate_quantized_model(
    original_model_path: str,
    quantized_model_path: str,
    test_data: np.ndarray,
    tolerance: float = 1e-3,
) -> Dict[str, float]:
    """
    Validate quantized model outputs match original within tolerance.
    
    Args:
        original_model_path: Path to original FP32 model
        quantized_model_path: Path to quantized INT8 model
        test_data: Sample input data for validation
        tolerance: Maximum allowed difference (default 1e-3)
    
    Returns:
        Dictionary with validation metrics (max_diff, mean_diff, accuracy_maintained)
    """
    # Load both models
    original_session = load_optimized_onnx_model(original_model_path)
    quantized_session = load_optimized_onnx_model(quantized_model_path)
    
    # Run inference on both
    input_name = original_session.get_inputs()[0].name
    original_output = original_session.run(None, {input_name: test_data})[0]
    quantized_output = quantized_session.run(None, {input_name: test_data})[0]
    
    # Compare outputs
    diff = np.abs(original_output - quantized_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Check predictions match
    original_preds = np.argmax(original_output, axis=1)
    quantized_preds = np.argmax(quantized_output, axis=1)
    accuracy = np.mean(original_preds == quantized_preds)
    
    validation_results = {
        "max_difference": float(max_diff),
        "mean_difference": float(mean_diff),
        "prediction_accuracy": float(accuracy),
        "within_tolerance": max_diff <= tolerance,
    }
    
    logger.info(f"Quantization validation results:")
    logger.info(f"  Max difference: {max_diff:.6f}")
    logger.info(f"  Mean difference: {mean_diff:.6f}")
    logger.info(f"  Prediction match rate: {accuracy:.2%}")
    logger.info(f"  Within tolerance: {validation_results['within_tolerance']}")
    
    return validation_results
```

**Quantization Trade-offs:**

| Aspect | FP32 | INT8 | Notes |
|--------|------|------|-------|
| Model Size | 100% | 25% | 4x reduction |
| Inference Speed | 1x | 2-4x | Hardware dependent |
| Accuracy | Baseline | 99-99.5% | Minimal degradation |
| Memory | High | Low | Better for deployment |

**When to Use:**
- ✅ **Recommended**: Production models where 0.5-1% accuracy drop is acceptable
- ✅ **Recommended**: Memory-constrained environments
- ⚠️ **Validate First**: Mission-critical applications (verify accuracy on test set)
- ❌ **Avoid**: Research/experimentation where maximum accuracy is critical

### Phase 4: Hardware-Specific Acceleration (MEDIUM IMPACT)

**Expected Speedup: 1.2-1.5x additional (28ms → 20-23ms)**

#### GPU: TensorRT Execution Provider

```python
def configure_tensorrt_provider() -> List[Tuple[str, Dict]]:
    """
    Configure TensorRT execution provider for optimal GPU performance.
    
    TensorRT provides:
    - Layer fusion and kernel auto-tuning
    - FP16 precision support (2x faster)
    - Dynamic tensor memory management
    - Optimal kernel selection per GPU architecture
    
    Returns:
        List of configured execution providers
    """
    providers = []
    
    if torch.cuda.is_available():
        # TensorRT configuration
        trt_config = {
            # Precision
            "trt_fp16_enable": True,  # Enable FP16 precision (2x speedup)
            
            # Engine caching (faster cold starts)
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "/tmp/trt_cache",
            
            # Memory management
            "trt_max_workspace_size": 2 * 1024**3,  # 2GB workspace
            "trt_max_partition_iterations": 1000,
            
            # Optimization
            "trt_min_subgraph_size": 3,  # Minimum ops to offload to TensorRT
            "trt_dla_enable": False,  # Deep Learning Accelerator (Jetson only)
            
            # Timing cache (faster engine builds)
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": "/tmp/trt_timing_cache",
        }
        
        providers.append(("TensorrtExecutionProvider", trt_config))
        logger.info("✓ Configured TensorRT execution provider")
    
    return providers


def configure_cuda_provider() -> List[Tuple[str, Dict]]:
    """
    Configure CUDA execution provider as fallback.
    
    Returns:
        List of configured execution providers
    """
    providers = []
    
    if torch.cuda.is_available():
        cuda_config = {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 2 * 1024**3,  # 2GB memory limit
            "cudnn_conv_algo_search": "EXHAUSTIVE",  # Find best conv algorithm
            "do_copy_in_default_stream": True,
        }
        
        providers.append(("CUDAExecutionProvider", cuda_config))
        logger.info("✓ Configured CUDA execution provider")
    
    return providers
```

#### CPU: Thread Affinity and SIMD Optimization

```python
def configure_cpu_optimization(
    num_physical_cores: int = None,
) -> ort.SessionOptions:
    """
    Configure CPU-specific optimizations.
    
    Args:
        num_physical_cores: Number of physical CPU cores (auto-detect if None)
    
    Returns:
        Configured SessionOptions
    """
    if num_physical_cores is None:
        num_physical_cores = os.cpu_count() // 2  # Assume hyperthreading
    
    sess_options = ort.SessionOptions()
    
    # Thread configuration for CPU
    sess_options.intra_op_num_threads = num_physical_cores  # Parallel within ops
    sess_options.inter_op_num_threads = 1  # Sequential operator execution
    
    # Enable CPU-specific optimizations
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Memory optimizations
    sess_options.enable_cpu_mem_arena = True
    sess_options.enable_mem_pattern = True
    
    logger.info(f"✓ Configured CPU optimization for {num_physical_cores} cores")
    
    return sess_options
```

### Complete Integration Example

```python
class OptimizedONNXInferenceHandler:
    """
    Production-ready ONNX inference handler with all optimizations applied.
    """
    
    def __init__(
        self,
        model_dir: str,
        enable_bert_fusion: bool = True,
        enable_quantization: bool = False,
        enable_tensorrt: bool = True,
    ):
        """
        Initialize optimized ONNX inference handler.
        
        Args:
            model_dir: Directory containing ONNX model
            enable_bert_fusion: Apply BERT-specific optimizations
            enable_quantization: Use INT8 quantized model
            enable_tensorrt: Use TensorRT execution provider (GPU only)
        """
        self.model_dir = Path(model_dir)
        self.session = None
        
        # Select model variant
        if enable_quantization:
            model_path = self.model_dir / "model_quantized.onnx"
            if not model_path.exists():
                logger.info("Quantized model not found, quantizing original...")
                self._quantize_model()
        elif enable_bert_fusion:
            model_path = self.model_dir / "model_optimized.onnx"
            if not model_path.exists():
                logger.info("Optimized model not found, applying BERT fusion...")
                self._optimize_bert_model()
        else:
            model_path = self.model_dir / "model.onnx"
        
        # Configure execution providers
        providers = []
        if enable_tensorrt and torch.cuda.is_available():
            providers.extend(configure_tensorrt_provider())
        if torch.cuda.is_available():
            providers.extend(configure_cuda_provider())
        providers.append("CPUExecutionProvider")
        
        # Create session with optimizations
        sess_options = configure_cpu_optimization() if not torch.cuda.is_available() else ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )
        
        logger.info(f"✓ Initialized optimized ONNX inference handler")
        logger.info(f"  Model: {model_path.name}")
        logger.info(f"  Providers: {self.session.get_providers()}")
    
    def _optimize_bert_model(self):
        """Apply BERT fusion optimizations."""
        original = self.model_dir / "model.onnx"
        optimized = self.model_dir / "model_optimized.onnx"
        optimize_bert_model(str(original), str(optimized))
    
    def _quantize_model(self):
        """Apply INT8 quantization."""
        original = self.model_dir / "model.onnx"
        quantized = self.model_dir / "model_quantized.onnx"
        quantize_model(str(original), str(quantized))
    
    def predict(self, input_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Run optimized inference.
        
        Args:
            input_data: Dictionary of input tensors
        
        Returns:
            Model predictions
        """
        return self.session.run(None, input_data)[0]


# Usage example
def model_fn(model_dir, context=None):
    """Updated model loading function with optimizations."""
    # ... existing code ...
    
    # Load optimized ONNX model
    handler = OptimizedONNXInferenceHandler(
        model_dir=model_dir,
        enable_bert_fusion=True,  # Enable BERT optimizations
        enable_quantization=False,  # Disable quantization initially (validate first)
        enable_tensorrt=True,  # Enable TensorRT on GPU
    )
    
    return {
        "model": handler.session,  # Use optimized session
        "config": config,
        # ... other artifacts ...
    }
```

## Integration with Inference Handler

### Overview

The ONNX optimization functions integrate seamlessly with the existing PyTorch inference handler (`pytorch_inference_handler.py`). The integration is designed to be:

1. **Backward Compatible**: Existing deployments continue working without changes
2. **Opt-In**: Optimizations are enabled via configuration, not forced
3. **Graceful Fallback**: Automatically falls back to baseline if optimization fails
4. **Zero-Downtime**: Can be deployed with existing models, optimizes on first load

### Current Model Loading Flow

From `projects/rnr_pytorch_bedrock/dockers/pytorch_inference_handler.py`:

```python
def model_fn(model_dir, context=None):
    """Current implementation - loads ONNX with minimal configuration"""
    model_filename = "model.pth"
    model_artifact_name = "model_artifacts.pth"
    onnx_model_path = os.path.join(model_dir, "model.onnx")
    
    # Load config and artifacts
    load_config, embedding_mat, vocab, model_class = load_artifacts(
        os.path.join(model_dir, model_artifact_name), device_l=device
    )
    
    # Load model based on file type
    if os.path.exists(onnx_model_path):
        logger.info("Detected ONNX model.")
        # ❌ CURRENT: Uses basic load_onnx_model (Phase 1 only)
        model = load_onnx_model(onnx_model_path)
    else:
        logger.info("Detected PyTorch model.")
        model = load_model(...)
    
    # ... rest of initialization ...
    
    return {
        "model": model,
        "config": config,
        # ... other artifacts ...
    }
```

**Current Behavior:**
- Uses `load_onnx_model(onnx_path)` - Phase 1 SessionOptions only
- No BERT fusion applied (Phase 2)
- No model optimization on cold start
- Expected latency: **335-545ms per request**

### Optimized Model Loading Flow (Phase 1 + Phase 2)

**Proposed Update** to enable full BERT optimization:

```python
def model_fn(model_dir, context=None):
    """Updated implementation - loads ONNX with Phase 1 + Phase 2 optimizations"""
    model_filename = "model.pth"
    model_artifact_name = "model_artifacts.pth"
    onnx_model_path = os.path.join(model_dir, "model.onnx")
    
    # Load config and artifacts (unchanged)
    load_config, embedding_mat, vocab, model_class = load_artifacts(
        os.path.join(model_dir, model_artifact_name), device_l=device
    )
    
    config = Config(**load_config)
    
    # Load model based on file type
    if os.path.exists(onnx_model_path):
        logger.info("Detected ONNX model.")
        
        # ✅ NEW: Use load_bert_optimized_model for Phase 1 + Phase 2
        # This function handles:
        # 1. Checking for pre-optimized model (model_optimized.onnx)
        # 2. Applying BERT fusion if needed (one-time, 10-30s)
        # 3. Loading with SessionOptions optimization (Phase 1)
        model = load_bert_optimized_model(
            model_dir=model_dir,
            enable_profiling=False,  # Disable in production
            inter_op_threads=1,      # Sequential operator execution
            intra_op_threads=4,      # Parallel within operators
        )
    else:
        logger.info("Detected PyTorch model.")
        model = load_model(...)
    
    # ... rest of initialization (unchanged) ...
    
    return {
        "model": model,  # Now uses optimized ONNX session
        "config": config,
        # ... other artifacts ...
    }
```

**Key Change:**
Replace `load_onnx_model(onnx_path)` → `load_bert_optimized_model(model_dir)`

**New Behavior:**
- Applies Phase 1 + Phase 2 optimizations automatically
- One-time BERT fusion on first cold start (10-30 seconds)
- Subsequent loads use cached `model_optimized.onnx` (fast)
- Expected latency: **55-110ms per request** (3-6x faster)

### Cold Start vs Warm Start Behavior

#### First Deployment (Cold Start)

**Sequence of operations when model_fn is called for the first time:**

```
1. Model Directory Structure:
   /opt/ml/model/
   ├── model.onnx                  ← Original ONNX model
   ├── model_artifacts.pth         ← Config, embeddings, vocab
   ├── hyperparameters.json        ← Training hyperparameters
   └── feature_columns.txt         ← Feature ordering

2. load_bert_optimized_model() is called:
   ├─ Check for model_optimized.onnx
   │  └─ NOT FOUND ❌
   │
   ├─ Call optimize_bert_model()
   │  ├─ Load original model.onnx
   │  ├─ Apply BERT fusion transformations
   │  │  ├─ Fuse multi-head attention blocks
   │  │  ├─ Fuse LayerNorm operations
   │  │  ├─ Fuse GELU activations
   │  │  ├─ Fuse skip connections
   │  │  └─ Fuse embedding layers
   │  ├─ Save optimized model → model_optimized.onnx
   │  └─ [10-30 seconds one-time cost]
   │
   └─ Load model_optimized.onnx with SessionOptions
      ├─ GraphOptimizationLevel.ORT_ENABLE_ALL
      ├─ ExecutionMode.ORT_SEQUENTIAL
      ├─ Memory optimizations enabled
      └─ Thread configuration tuned

3. Final Model Directory:
   /opt/ml/model/
   ├── model.onnx                  ← Original (preserved)
   ├── model_optimized.onnx        ← NEW: Optimized model ✅
   ├── model_artifacts.pth
   ├── hyperparameters.json
   └── feature_columns.txt

4. First Request Latency:
   Cold start: ~30 seconds (one-time BERT fusion)
   + First inference: ~55-110ms
   = Total first request: ~30 seconds
```

#### Subsequent Requests (Warm Start)

**After the first cold start, all future loads are fast:**

```
1. Model Directory Structure:
   /opt/ml/model/
   ├── model.onnx                  ← Original
   ├── model_optimized.onnx        ← Cached optimized model ✅
   ├── model_artifacts.pth
   ├── hyperparameters.json
   └── feature_columns.txt

2. load_bert_optimized_model() is called:
   ├─ Check for model_optimized.onnx
   │  └─ FOUND ✅ (cached from cold start)
   │
   └─ Load model_optimized.onnx with SessionOptions
      └─ [< 1 second, no optimization needed]

3. Request Latency:
   Model load: < 1 second
   + Inference: ~55-110ms
   = Total: ~55-110ms (fast!) ✅
```

**Key Insight**: The 10-30 second BERT fusion overhead occurs **once** per model deployment. All subsequent requests benefit from the cached optimized model.

### Graceful Fallback Mechanism

The implementation includes automatic fallback to ensure production stability:

```python
def load_bert_optimized_model(
    model_dir: str,
    enable_profiling: bool = False,
    inter_op_threads: int = 1,
    intra_op_threads: int = 4,
) -> ort.InferenceSession:
    """
    Load BERT-optimized ONNX model with graceful fallback.
    
    Fallback Chain:
    1. Try load model_optimized.onnx with SessionOptions (Phase 1 + 2)
    2. If not found, try optimize model.onnx with BERT fusion (Phase 2)
    3. If fusion fails, fall back to model.onnx with SessionOptions (Phase 1)
    4. If SessionOptions fail, fall back to basic load_onnx_model (baseline)
    """
    original_model = Path(model_dir) / "model.onnx"
    optimized_model = Path(model_dir) / "model_optimized.onnx"
    
    # === Attempt 1: Load pre-optimized model (fastest) ===
    if optimized_model.exists():
        try:
            logger.info(f"Loading pre-optimized model: {optimized_model}")
            return load_optimized_onnx_model(
                onnx_path=optimized_model,
                enable_profiling=enable_profiling,
                inter_op_threads=inter_op_threads,
                intra_op_threads=intra_op_threads,
            )
        except Exception as e:
            logger.warning(f"Failed to load pre-optimized model: {e}")
            logger.warning("Falling back to original model with optimization...")
    
    # === Attempt 2: Optimize original model with BERT fusion ===
    try:
        logger.info("Pre-optimized model not found, applying BERT fusion...")
        optimize_bert_model(
            input_model_path=str(original_model),
            output_model_path=str(optimized_model),
            model_type="bert",
            num_heads=12,
            hidden_size=768,
        )
        
        # Load newly optimized model
        return load_optimized_onnx_model(
            onnx_path=optimized_model,
            enable_profiling=enable_profiling,
            inter_op_threads=inter_op_threads,
            intra_op_threads=intra_op_threads,
        )
    except Exception as e:
        logger.warning(f"BERT fusion optimization failed: {e}")
        logger.warning("Falling back to Phase 1 optimizations only...")
    
    # === Attempt 3: Load original model with Phase 1 optimizations ===
    try:
        logger.info("Loading original model with SessionOptions (Phase 1 only)...")
        return load_optimized_onnx_model(
            onnx_path=original_model,
            enable_profiling=enable_profiling,
            inter_op_threads=inter_op_threads,
            intra_op_threads=intra_op_threads,
        )
    except Exception as e:
        logger.error(f"Phase 1 optimization failed: {e}")
        logger.error("Falling back to baseline configuration...")
    
    # === Attempt 4: Load with baseline configuration (no optimizations) ===
    logger.warning("⚠️  Using baseline ONNX configuration (no optimizations)")
    return load_onnx_model(str(original_model))
```

**Fallback Levels:**

| Level | Configuration | Expected Latency | Status |
|-------|--------------|------------------|--------|
| **Level 1** | Phase 1 + Phase 2 (model_optimized.onnx + SessionOptions) | 55-110ms | ✅ Best |
| **Level 2** | Phase 1 only (model.onnx + SessionOptions) | 110-170ms | ⚠️ Good |
| **Level 3** | Baseline (model.onnx, no optimizations) | 335-545ms | ❌ Baseline |

**Production Safety:**
- Each fallback level is tested independently
- Errors are logged with clear diagnostics
- Service remains operational even if optimization fails
- Monitoring alerts can trigger investigation of fallback usage

### Configuration Options

The optimization can be controlled via environment variables for flexible deployment:

```python
# In pytorch_inference_handler.py

def model_fn(model_dir, context=None):
    """Model loading with configurable optimization level"""
    
    # Read optimization configuration from environment
    enable_bert_fusion = os.environ.get("ENABLE_BERT_FUSION", "true").lower() == "true"
    enable_profiling = os.environ.get("ENABLE_ONNX_PROFILING", "false").lower() == "true"
    inter_op_threads = int(os.environ.get("ONNX_INTER_OP_THREADS", "1"))
    intra_op_threads = int(os.environ.get("ONNX_INTRA_OP_THREADS", "4"))
    
    # ... load config and artifacts ...
    
    # Load ONNX model with configured optimizations
    if os.path.exists(onnx_model_path):
        logger.info("Detected ONNX model.")
        
        if enable_bert_fusion:
            # Phase 1 + Phase 2: Full optimization
            logger.info("Loading with Phase 1 + Phase 2 optimizations (BERT fusion)")
            model = load_bert_optimized_model(
                model_dir=model_dir,
                enable_profiling=enable_profiling,
                inter_op_threads=inter_op_threads,
                intra_op_threads=intra_op_threads,
            )
        else:
            # Phase 1 only: SessionOptions
            logger.info("Loading with Phase 1 optimizations only (SessionOptions)")
            model = load_optimized_onnx_model(
                onnx_path=onnx_model_path,
                enable_profiling=enable_profiling,
                inter_op_threads=inter_op_threads,
                intra_op_threads=intra_op_threads,
            )
    else:
        logger.info("Detected PyTorch model.")
        model = load_model(...)
    
    # ... rest of initialization ...
```

**Environment Variable Reference:**

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `ENABLE_BERT_FUSION` | `true` | Enable Phase 2 BERT fusion optimization | `true`, `false` |
| `ENABLE_ONNX_PROFILING` | `false` | Enable ONNX Runtime profiling (debug only) | `true`, `false` |
| `ONNX_INTER_OP_THREADS` | `1` | Threads for parallel operator execution | `1`, `2`, `4` |
| `ONNX_INTRA_OP_THREADS` | `4` | Threads within operators (matrix ops) | `2`, `4`, `8` |

**Deployment Scenarios:**

```bash
# Production (recommended): Phase 1 + Phase 2
export ENABLE_BERT_FUSION=true
export ONNX_INTER_OP_THREADS=1
export ONNX_INTRA_OP_THREADS=4

# Debugging: Phase 1 + Phase 2 + Profiling
export ENABLE_BERT_FUSION=true
export ENABLE_ONNX_PROFILING=true

# Conservative: Phase 1 only (no BERT fusion)
export ENABLE_BERT_FUSION=false
export ONNX_INTER_OP_THREADS=1
export ONNX_INTRA_OP_THREADS=4

# Baseline (no optimizations, for comparison)
# Use original load_onnx_model() function
```

### Deployment Checklist

**Before deploying optimized inference handler:**

1. **✅ Code Changes**
   - [ ] Update `model_fn()` to call `load_bert_optimized_model()`
   - [ ] Verify `load_optimized_onnx_model()` is in `pl_train.py`
   - [ ] Verify `optimize_bert_model()` is in `pl_train.py`
   - [ ] Add environment variable configuration (optional)

2. **✅ Testing**
   - [ ] Test cold start behavior (first model load)
   - [ ] Test warm start behavior (subsequent loads)
   - [ ] Verify fallback mechanism works
   - [ ] Measure latency improvements
   - [ ] Validate accuracy matches baseline

3. **✅ Monitoring**
   - [ ] Add CloudWatch metrics for optimization success/failure
   - [ ] Add latency percentile tracking (p50, p95, p99)
   - [ ] Add alert for fallback to baseline configuration
   - [ ] Add alert for p95 latency > threshold

4. **✅ Rollback Plan**
   - [ ] Keep backup of original `pytorch_inference_handler.py`
   - [ ] Document rollback procedure
   - [ ] Test rollback in staging
   - [ ] Prepare hotfix branch if needed

### Integration Testing

**Test Plan for Optimized Inference:**

```python
def test_optimized_inference_integration():
    """Integration test for optimized ONNX inference"""
    
    # 1. Test cold start (BERT fusion)
    logger.info("Test 1: Cold start with BERT fusion")
    model_artifacts = model_fn(model_dir="/opt/ml/model")
    assert Path("/opt/ml/model/model_optimized.onnx").exists()
    
    # 2. Test warm start (cached optimized model)
    logger.info("Test 2: Warm start with cached model")
    model_artifacts = model_fn(model_dir="/opt/ml/model")
    # Should load in < 1 second
    
    # 3. Test inference latency
    logger.info("Test 3: Inference latency")
    test_input = create_test_input()
    
    latencies = []
    for _ in range(100):
        start = time.time()
        prediction = predict_fn(test_input, model_artifacts)
        latencies.append((time.time() - start) * 1000)
    
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    
    logger.info(f"p50 latency: {p50_latency:.2f}ms")
    logger.info(f"p95 latency: {p95_latency:.2f}ms")
    
    # Assert performance improvement
    assert p50_latency < 120, f"p50 latency {p50_latency}ms exceeds target 120ms"
    assert p95_latency < 180, f"p95 latency {p95_latency}ms exceeds target 180ms"
    
    # 4. Test accuracy matches baseline
    logger.info("Test 4: Accuracy validation")
    baseline_predictions = run_baseline_model(test_input)
    optimized_predictions = prediction["calibrated_predictions"]
    
    accuracy_match = np.allclose(baseline_predictions, optimized_predictions, atol=1e-3)
    assert accuracy_match, "Optimized predictions don't match baseline"
    
    logger.info("✅ All integration tests passed")
```

### Production Rollout Strategy

**Recommended Phased Rollout:**

```
Phase 1: Canary Deployment (Week 1)
├─ Deploy to 5% of production traffic
├─ Monitor latency and error rates
├─ Validate accuracy matches baseline
└─ Decision: Proceed or rollback

Phase 2: Gradual Rollout (Week 2)
├─ 5% → 25% → 50% → 100% traffic
├─ Monitor each step for 24 hours
├─ Compare metrics with baseline
└─ Keep baseline endpoint as fallback

Phase 3: Optimization (Week 3)
├─ All traffic on optimized endpoint
├─ Remove baseline endpoint
├─ Tune thread configuration if needed
└─ Document final configuration
```

### Troubleshooting Guide

**Common Issues and Solutions:**

| Issue | Symptom | Solution |
|-------|---------|----------|
| **BERT fusion fails** | "BERT fusion optimization failed" in logs | Check ONNX model has BERT layers, verify ONNX Runtime version |
| **OOM during fusion** | OutOfMemoryError during cold start | Increase instance memory or disable BERT fusion |
| **Latency regression** | p95 > baseline | Check thread configuration, verify optimized model loaded |
| **Accuracy drop** | Predictions differ from baseline | Disable BERT fusion, validate model export |
| **Slow cold start** | First request > 60 seconds | Expected for BERT fusion, consider pre-warming |

**Debug Mode:**

```python
# Enable detailed logging for troubleshooting
import logging
logging.getLogger("onnxruntime").setLevel(logging.DEBUG)
logging.getLogger("transformers").setLevel(logging.DEBUG)

# Enable ONNX Runtime profiling
export ENABLE_ONNX_PROFILING=true

# Profile output location
ls -la /tmp/onnx_profile*.json
```

## Performance Benchmarks

### Expected Performance Improvements

| Optimization | Baseline | Speedup | Latency | Cumulative |
|--------------|----------|---------|---------|------------|
| **Baseline** | 400ms | 1.0x | 400ms | - |
| **Phase 1: SessionOptions** | 400ms | 2.5x | 160ms | 2.5x |
| **Phase 2: BERT Fusion** | 160ms | 1.7x | 94ms | 4.3x |
| **Phase 3: INT8 Quantization** | 94ms | 1.6x | 59ms | 6.8x |
| **Phase 4: TensorRT** | 59ms | 1.3x | 45ms | 8.9x |

**Target: 50-100ms per request (4-10x speedup)**

### Validation Approach

For each optimization phase:

1. **Latency Testing**: Measure p50, p95, p99 latencies
2. **Accuracy Validation**: Compare predictions with baseline
3. **Throughput Testing**: Measure requests/second capacity
4. **Resource Monitoring**: CPU/GPU/memory utilization

```python
def benchmark_optimization(
    model_path: str,
    test_data: List[Dict[str, np.ndarray]],
    num_iterations: int = 100,
) -> Dict[str, float]:
    """
    Benchmark ONNX model performance.
    
    Args:
        model_path: Path to ONNX model
        test_data: Sample inputs for benchmarking
        num_iterations: Number of benchmark iterations
    
    Returns:
        Dictionary with performance metrics
    """
    session = load_optimized_onnx_model(model_path)
    latencies = []
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, test_data[0])
    
    # Benchmark
    for i in range(num_iterations):
        start = time.time()
        _ = session.run(None, test_data[i % len(test_data)])
        latencies.append((time.time() - start) * 1000)
    
    return {
        "p50_latency_ms": np.percentile(latencies, 50),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
        "mean_latency_ms": np.mean(latencies),
        "std_latency_ms": np.std(latencies),
    }
```

## Implementation Roadmap

### Week 1: Phase 1 Implementation
- Implement `load_optimized_onnx_model` with SessionOptions
- Deploy to staging environment
- Measure latency improvements
- **Expected Result**: 2-3x speedup (400ms → 160ms)

### Week 2: Phase 2 Implementation  
- Implement BERT fusion optimizer
- Generate optimized ONNX models
- A/B test optimized vs baseline
- **Expected Result**: Additional 1.5-2x speedup (160ms → 94ms)

### Week 3: Phase 3 Validation
- Generate INT8 quantized models
- Validate accuracy on test set
- Benchmark quantized performance
- **Decision Point**: Deploy quantization if accuracy maintained

### Week 4: Phase 4 (GPU Only)
- Configure TensorRT execution provider
- Benchmark TensorRT performance
- Deploy to production GPU instances
- **Expected Result**: Additional 1.2-1.5x speedup

## Risk Mitigation

### Accuracy Degradation Risk

**Mitigation Strategy:**
1. Validate each optimization phase independently
2. Compare predictions with baseline on large test set
3. Monitor production metrics (precision, recall, F1)
4. Implement rollback mechanism if accuracy drops

### Performance Regression Risk

**Mitigation Strategy:**
1. A/B test optimizations before full rollout
2. Monitor p99 latencies in production
3. Keep baseline model as fallback
4. Implement canary deployments

### Hardware Compatibility Risk

**Mitigation Strategy:**
1. Test on all target instance types (CPU/GPU)
2. Graceful fallback to CPU provider
3. Document minimum hardware requirements
4. Version lock ONNX Runtime dependencies

## Monitoring and Observability

### Key Metrics to Track

```python
class ONNXInferenceMetrics:
    """Metrics collection for ONNX inference monitoring."""
    
    def __init__(self):
        self.latency_histogram = []
        self.provider_usage = {}
        self.optimization_level = None
    
    def record_inference(
        self,
        latency_ms: float,
        provider: str,
        success: bool,
    ):
        """Record inference metrics."""
        self.latency_histogram.append(latency_ms)
        self.provider_usage[provider] = self.provider_usage.get(provider, 0) + 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "p50_latency_ms": np.percentile(self.latency_histogram, 50),
            "p95_latency_ms": np.percentile(self.latency_histogram, 95),
            "p99_latency_ms": np.percentile(self.latency_histogram, 99),
            "provider_distribution": self.provider_usage,
            "total_requests": len(self.latency_histogram),
        }
```

### Production Alerts

1. **Latency Degradation**: Alert if p95 latency > 150ms
2. **Accuracy Drop**: Alert if F1 score < baseline - 1%
3. **Provider Fallback**: Alert if TensorRT fallback rate > 10%
4. **Memory Issues**: Alert if OOM errors detected

## Success Criteria

### Phase 1 Success Criteria
- ✅ p50 latency < 170ms (from 400ms baseline)
- ✅ No accuracy degradation (< 0.1% difference)
- ✅ Successful deployment to staging

### Phase 2 Success Criteria
- ✅ p50 latency < 100ms (additional 1.5x improvement)
- ✅ BERT fusion applied successfully
- ✅ No increase in error rate

### Phase 3 Success Criteria (Optional)
- ✅ p50 latency < 60ms (additional 1.5x improvement)
- ✅ Accuracy maintained (> 99% of baseline)
- ✅ Model size reduced by 4x

### Phase 4 Success Criteria (GPU Only)
- ✅ p50 latency < 50ms (additional 1.2x improvement)
- ✅ TensorRT engine cached successfully
- ✅ No increase in GPU memory usage

## Conclusion

This comprehensive optimization strategy addresses the root causes of high ONNX inference latency through a phased approach:

1. **SessionOptions Configuration**: Quick win with 2-3x speedup
2. **BERT Fusion**: Model-specific optimizations for 1.5-2x additional speedup
3. **INT8 Quantization**: Optional 1.5-2x speedup with accuracy validation
4. **Hardware Acceleration**: Final 1.2-1.5x speedup on GPU

**Total Expected Improvement: 4-10x speedup (400ms → 50-100ms)**

The phased rollout with validation at each stage ensures production stability while achieving significant performance improvements. Each phase can be deployed independently, allowing for iterative optimization and risk mitigation.

## References

1. [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/tune-performance.html)
2. [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
3. [Transformer Model Optimization](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/transformers)
4. [Dynamic Quantization Guide](https://onnxruntime.ai/docs/performance/quantization.html)
5. [TensorRT Execution Provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)

## Appendix: Configuration Examples

### Example 1: CPU Production Configuration

```python
# Optimized for CPU-only inference
session = load_optimized_onnx_model(
    onnx_path="model_optimized.onnx",
    enable_profiling=False,
    inter_op_threads=1,  # Sequential operator execution
    intra_op_threads=4,  # Parallel within operators
)
```

### Example 2: GPU Production Configuration

```python
# Optimized for GPU inference with TensorRT
handler = OptimizedONNXInferenceHandler(
    model_dir="/opt/ml/model",
    enable_bert_fusion=True,
    enable_quantization=False,
    enable_tensorrt=True,
)
```

### Example 3: Memory-Constrained Configuration

```python
# Optimized for memory-constrained environments
handler = OptimizedONNXInferenceHandler(
    model_dir="/opt/ml/model",
    enable_bert_fusion=True,
    enable_quantization=True,  # 4x memory reduction
    enable_tensorrt=False,
)
```
