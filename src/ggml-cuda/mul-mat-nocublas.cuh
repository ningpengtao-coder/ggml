#pragma once

#include "common.cuh"

// 不依赖 cuBLAS 的 GEMM 实现（GGML_CUDA_NO_CUBLAS）。
//
// 统一形式：C[m][n] = sum_k A[m*a_m + k*a_k] * B[n*b_n + k*b_k]，
// C 的地址为 C[n*c_n + m*c_m]。步长均以元素个数计，这样 mul_mat、
// out_prod、conv3d、convrot 这些不同内存布局都能复用同一套 tile 内核。

void ggml_cuda_mul_mat_slice_nocublas(
    const void * A, ggml_type type_a, int64_t a_m, int64_t a_k,
    const void * B, ggml_type type_b, int64_t b_n, int64_t b_k,
    void * C, ggml_type type_c, int64_t c_m, int64_t c_n,
    int64_t M, int64_t N, int64_t K, bool accumulate, cudaStream_t stream);

// int8 x int8 -> int32，用于 audio.cpp 的 CONVROT_LINEAR 量化线性层。
// 布局固定为 A[m*K + k]、B[n*K + k]、C[n*ldc + m]。
void ggml_cuda_mul_mat_i8_i32_nocublas(
    const int8_t * A, const int8_t * B, int32_t * C,
    int64_t M, int64_t N, int64_t K, int64_t ldc, bool accumulate, cudaStream_t stream);
