# VN97 R2-K2C — PyTorch/ORT backend matrix

K2B proved that both the PyTorch-to-ORT-CPU and ORT-CPU-to-ORT-CUDA
comparisons show measurable FP16 drift. However, the K2B PyTorch reference was
itself executed on CUDA. That means the PyTorch->ORT CPU comparison still
mixed two variables:

- graph/runtime lowering, and
- CUDA-vs-CPU backend numerics.

K2C separates those variables before any K2 acceptance threshold is defined.

## One-step diagnostic matrix

K2C runs one real token through the exact inherited Mamba-2 2.7B weights and
the exact G0.4 graph:

    PyTorch CPU
    PyTorch CUDA
    ORT CPU
    ORT CUDA

It records:

- PyTorch CUDA vs PyTorch CPU;
- PyTorch CPU vs ORT CPU;
- PyTorch CUDA vs ORT CUDA;
- ORT CPU vs ORT CUDA;
- exact greedy argmax equality for each comparison;
- per-layer convolution-state and SSD-state FP16 epsilon error;
- the first layer exceeding the existing two-epsilon K1I semantic reference.

The two-epsilon value is diagnostic only. K2C does not define a production
acceptance threshold and cannot authorize production.

Only one recurrent token is used because a full 2.7B PyTorch CPU pass is
expensive. K2C is an isolation experiment, not the final K2 validation suite.

## Interpretation

- PyTorch CUDA vs CPU wide, PyTorch CPU vs ORT CPU tight:
  the dominant drift is backend-dependent; do not rewrite ONNX semantics.
- PyTorch CPU vs ORT CPU wide:
  an ONNX lowering/runtime discrepancy remains and the first divergent layer
  becomes the target for K2D same-input stage tracing.
- ORT CPU vs CUDA wide:
  CUDAExecutionProvider adds a separate runtime envelope that must be measured
  independently after the CPU lowering boundary is understood.
- All boundaries tight:
  K2D can move directly to repeated-run calibration for a fail-closed K2 gate.

## Run

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2c_backend_matrix.sh

Expected evidence:

    /kaggle/working/vn97-k2c-backend-matrix.json
    /kaggle/working/VN97-R2-K2C-backend-matrix.zip
