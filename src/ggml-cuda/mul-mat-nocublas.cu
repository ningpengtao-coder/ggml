#include "mul-mat-nocublas.cuh"

#include "convert.cuh"

namespace {

constexpr int MM_TILE = 16;

__device__ __forceinline__ float mm_to_float(float x) { return x; }
__device__ __forceinline__ float mm_to_float(half x) { return __half2float(x); }
__device__ __forceinline__ float mm_to_float(nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ void mm_from_float(float v, float & out) { out = v; }
__device__ __forceinline__ void mm_from_float(float v, half & out) { out = __float2half(v); }

template <typename TA, typename TB, typename TC>
__global__ void mul_mat_slice_kernel(
        const TA * __restrict__ A, const int64_t a_m, const int64_t a_k,
        const TB * __restrict__ B, const int64_t b_n, const int64_t b_k,
        TC * __restrict__ C, const int64_t c_m, const int64_t c_n,
        const int64_t M, const int64_t N, const int64_t K,
        const bool accumulate) {
    __shared__ float As[MM_TILE][MM_TILE];
    __shared__ float Bs[MM_TILE][MM_TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int64_t row0 = static_cast<int64_t>(blockIdx.y) * MM_TILE;
    const int64_t col0 = static_cast<int64_t>(blockIdx.x) * MM_TILE;

    float acc = 0.0f;
    for (int64_t k0 = 0; k0 < K; k0 += MM_TILE) {
        const int64_t kk = k0 + ty;
        As[tx][ty] = (row0 + tx < M && kk < K)
            ? mm_to_float(A[(row0 + tx)*a_m + kk*a_k]) : 0.0f;
        Bs[tx][ty] = (col0 + tx < N && kk < K)
            ? mm_to_float(B[(col0 + tx)*b_n + kk*b_k]) : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < MM_TILE; ++k) {
            acc += As[tx][k] * Bs[ty][k];
        }
        __syncthreads();
    }

    if (row0 + tx < M && col0 + ty < N) {
        TC * out = &C[(col0 + ty)*c_n + (row0 + tx)*c_m];
        if (accumulate) {
            acc += mm_to_float(*out);
        }
        mm_from_float(acc, *out);
    }
}

__global__ void mul_mat_i8_i32_kernel(
        const int8_t * __restrict__ A, const int64_t lda,
        const int8_t * __restrict__ B, const int64_t ldb,
        int32_t * __restrict__ C, const int64_t ldc,
        const int64_t M, const int64_t N, const int64_t K,
        const bool accumulate) {
    __shared__ int As[MM_TILE][MM_TILE];
    __shared__ int Bs[MM_TILE][MM_TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int64_t row0 = static_cast<int64_t>(blockIdx.y) * MM_TILE;
    const int64_t col0 = static_cast<int64_t>(blockIdx.x) * MM_TILE;

    int32_t acc = 0;
    for (int64_t k0 = 0; k0 < K; k0 += MM_TILE) {
        const int64_t kk = k0 + ty;
        As[tx][ty] = (row0 + tx < M && kk < K) ? static_cast<int>(A[(row0 + tx)*lda + kk]) : 0;
        Bs[tx][ty] = (col0 + tx < N && kk < K) ? static_cast<int>(B[(col0 + tx)*ldb + kk]) : 0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < MM_TILE; ++k) {
            acc += As[tx][k] * Bs[ty][k];
        }
        __syncthreads();
    }

    if (row0 + tx < M && col0 + ty < N) {
        int32_t * out = &C[(col0 + ty)*ldc + (row0 + tx)];
        if (accumulate) {
            acc += *out;
        }
        *out = acc;
    }
}

template <typename TA, typename TB, typename TC>
void launch_slice(
        const void * A, int64_t a_m, int64_t a_k,
        const void * B, int64_t b_n, int64_t b_k,
        void * C, int64_t c_m, int64_t c_n,
        int64_t M, int64_t N, int64_t K, bool accumulate, cudaStream_t stream) {
    const dim3 block(MM_TILE, MM_TILE);
    const dim3 grid(
        static_cast<unsigned>((N + MM_TILE - 1) / MM_TILE),
        static_cast<unsigned>((M + MM_TILE - 1) / MM_TILE));
    mul_mat_slice_kernel<TA, TB, TC><<<grid, block, 0, stream>>>(
        static_cast<const TA *>(A), a_m, a_k,
        static_cast<const TB *>(B), b_n, b_k,
        static_cast<TC *>(C), c_m, c_n,
        M, N, K, accumulate);
}

template <typename TB, typename TC>
void dispatch_a(
        ggml_type type_a,
        const void * A, int64_t a_m, int64_t a_k,
        const void * B, int64_t b_n, int64_t b_k,
        void * C, int64_t c_m, int64_t c_n,
        int64_t M, int64_t N, int64_t K, bool accumulate, cudaStream_t stream) {
    switch (type_a) {
        case GGML_TYPE_F32:
            launch_slice<float, TB, TC>(A, a_m, a_k, B, b_n, b_k, C, c_m, c_n, M, N, K, accumulate, stream);
            break;
        case GGML_TYPE_F16:
            launch_slice<half, TB, TC>(A, a_m, a_k, B, b_n, b_k, C, c_m, c_n, M, N, K, accumulate, stream);
            break;
        case GGML_TYPE_BF16:
            launch_slice<nv_bfloat16, TB, TC>(A, a_m, a_k, B, b_n, b_k, C, c_m, c_n, M, N, K, accumulate, stream);
            break;
        default:
            GGML_ABORT("ggml_cuda_mul_mat_slice_nocublas: unsupported A type %s", ggml_type_name(type_a));
    }
}

template <typename TC>
void dispatch_b(
        ggml_type type_a, ggml_type type_b,
        const void * A, int64_t a_m, int64_t a_k,
        const void * B, int64_t b_n, int64_t b_k,
        void * C, int64_t c_m, int64_t c_n,
        int64_t M, int64_t N, int64_t K, bool accumulate, cudaStream_t stream) {
    switch (type_b) {
        case GGML_TYPE_F32:
            dispatch_a<float, TC>(type_a, A, a_m, a_k, B, b_n, b_k, C, c_m, c_n, M, N, K, accumulate, stream);
            break;
        case GGML_TYPE_F16:
            dispatch_a<half, TC>(type_a, A, a_m, a_k, B, b_n, b_k, C, c_m, c_n, M, N, K, accumulate, stream);
            break;
        case GGML_TYPE_BF16:
            dispatch_a<nv_bfloat16, TC>(type_a, A, a_m, a_k, B, b_n, b_k, C, c_m, c_n, M, N, K, accumulate, stream);
            break;
        default:
            GGML_ABORT("ggml_cuda_mul_mat_slice_nocublas: unsupported B type %s", ggml_type_name(type_b));
    }
}

} // namespace

void ggml_cuda_mul_mat_slice_nocublas(
        const void * A, ggml_type type_a, int64_t a_m, int64_t a_k,
        const void * B, ggml_type type_b, int64_t b_n, int64_t b_k,
        void * C, ggml_type type_c, int64_t c_m, int64_t c_n,
        int64_t M, int64_t N, int64_t K, bool accumulate, cudaStream_t stream) {
    if (M <= 0 || N <= 0) {
        return;
    }

    switch (type_c) {
        case GGML_TYPE_F32:
            dispatch_b<float>(type_a, type_b, A, a_m, a_k, B, b_n, b_k, C, c_m, c_n, M, N, K, accumulate, stream);
            break;
        case GGML_TYPE_F16:
            dispatch_b<half>(type_a, type_b, A, a_m, a_k, B, b_n, b_k, C, c_m, c_n, M, N, K, accumulate, stream);
            break;
        default:
            GGML_ABORT("ggml_cuda_mul_mat_slice_nocublas: unsupported C type %s", ggml_type_name(type_c));
    }
}

void ggml_cuda_mul_mat_i8_i32_nocublas(
        const int8_t * A, const int8_t * B, int32_t * C,
        int64_t M, int64_t N, int64_t K, int64_t ldc, bool accumulate, cudaStream_t stream) {
    if (M <= 0 || N <= 0) {
        return;
    }

    const dim3 block(MM_TILE, MM_TILE);
    const dim3 grid(
        static_cast<unsigned>((N + MM_TILE - 1) / MM_TILE),
        static_cast<unsigned>((M + MM_TILE - 1) / MM_TILE));
    mul_mat_i8_i32_kernel<<<grid, block, 0, stream>>>(A, K, B, K, C, ldc, M, N, K, accumulate);
}
