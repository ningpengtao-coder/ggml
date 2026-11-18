#include "common.cuh"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always the state row size, set in-kernel)
struct ggml_cuda_ssm_scan_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots
};

void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
void ggml_cuda_op_ssm_scan_gated(
    ggml_backend_cuda_context & ctx, ggml_tensor * scan, ggml_tensor * scale, ggml_tensor * glu);
#endif

// same op, but writes the state snapshot(s) into the cache instead of dst (see ggml_cuda_try_ssm_scan_cache_fusion)
void ggml_cuda_op_ssm_scan_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                       ggml_cuda_ssm_scan_fused_cache cache);
