#include "scale.cuh"
#include "convert.cuh"

#define MAX_GRIDDIM_X 0x7FFFFFFF

// src and dst may be views: rows are contiguous, dims 1..3 follow the strides (in elements).
template <typename T>
static __global__ void scale_f32(const T * x, T * dst, const float scale, const float bias, const uint32_t k,
                                 const uint3 ne0, const uint3 ne1, const uint3 ne2,
                                 const uint32_t s01, const uint32_t s02, const uint32_t s03,
                                 const uint32_t s1,  const uint32_t s2,  const uint32_t s3) {
    ggml_cuda_pdl_lc();
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    ggml_cuda_pdl_sync();
    if (i >= k) {
        return;
    }

    const uint2 d0 = fast_div_modulo(i,    ne0); // <i / ne0, i0>
    const uint2 d1 = fast_div_modulo(d0.x, ne1); // <i / (ne0*ne1), i1>
    const uint2 d2 = fast_div_modulo(d1.x, ne2); // <i3, i2>

    const size_t i_src = d0.y + size_t(d1.y)*s01 + size_t(d2.y)*s02 + size_t(d2.x)*s03;
    const size_t i_dst = d0.y + size_t(d1.y)*s1  + size_t(d2.y)*s2  + size_t(d2.x)*s3;

    dst[i_dst] = ggml_cuda_cast<T>(scale * ggml_cuda_cast<float>(x[i_src]) + bias);
}

static __global__ void scale_f32_bcast(
        const float * x,
        float * dst,
        const float scale,
        const float bias,
        const int64_t ne0,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne03,
        const size_t s0,
        const size_t s1,
        const size_t s2,
        const size_t s3,
        const size_t s00,
        const size_t s01,
        const size_t s02,
        const size_t s03) {
    const int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;
    const int64_t n = ne0*ne1*ne2*ne3;

    for (int64_t i = tid; i < n; i += stride) {
        const int64_t i0 = i % ne0;
        const int64_t i1 = (i / ne0) % ne1;
        const int64_t i2 = (i / (ne0*ne1)) % ne2;
        const int64_t i3 = i / (ne0*ne1*ne2);
        const size_t i_dst = i3*s3 + i2*s2 + i1*s1 + i0*s0;
        const size_t i_src =
            (i3 % ne03)*s03 + (i2 % ne02)*s02 + (i1 % ne01)*s01 + (i0 % ne00)*s00;
        dst[i_dst] = scale * x[i_src] + bias;
    }
}

template <typename T>
static void scale_f32_cuda(const T * x, T * dst, const float scale, const float bias, const ggml_tensor * src0, const ggml_tensor * t, cudaStream_t stream) {
    const int64_t k  = ggml_nelements(src0);
    const size_t  ts = sizeof(T);
    GGML_ASSERT(k <= std::numeric_limits<uint32_t>::max());

    const uint3 ne0 = init_fastdiv_values(src0->ne[0]);
    const uint3 ne1 = init_fastdiv_values(src0->ne[1]);
    const uint3 ne2 = init_fastdiv_values(src0->ne[2]);

    const int64_t num_blocks = (k + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_f32<T>, launch_params, x, dst, scale, bias, (uint32_t) k, ne0, ne1, ne2,
        src0->nb[1]/ts, src0->nb[2]/ts, src0->nb[3]/ts,
        t->nb[1]/ts,    t->nb[2]/ts,    t->nb[3]/ts);
}

static void scale_f32_bcast_cuda(const ggml_tensor * src0, ggml_tensor * dst, const float scale, const float bias, cudaStream_t stream) {
    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 % sizeof(float) == 0);
    GGML_ASSERT(nb1 % sizeof(float) == 0);
    GGML_ASSERT(nb2 % sizeof(float) == 0);
    GGML_ASSERT(nb3 % sizeof(float) == 0);
    GGML_ASSERT(nb00 % sizeof(float) == 0);
    GGML_ASSERT(nb01 % sizeof(float) == 0);
    GGML_ASSERT(nb02 % sizeof(float) == 0);
    GGML_ASSERT(nb03 % sizeof(float) == 0);

    const size_t s0 = nb0 / sizeof(float);
    const size_t s1 = nb1 / sizeof(float);
    const size_t s2 = nb2 / sizeof(float);
    const size_t s3 = nb3 / sizeof(float);
    const size_t s00 = nb00 / sizeof(float);
    const size_t s01 = nb01 / sizeof(float);
    const size_t s02 = nb02 / sizeof(float);
    const size_t s03 = nb03 / sizeof(float);

    const int64_t nelements = ggml_nelements(dst);
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_f32_bcast<<<MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(
        (const float *) src0->data,
        (float *) dst->data,
        scale,
        bias,
        ne0,
        ne1,
        ne2,
        ne3,
        ne00,
        ne01,
        ne02,
        ne03,
        s0,
        s1,
        s2,
        s3,
        s00,
        s01,
        s02,
        s03);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == src0->type);
    GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst));
    GGML_ASSERT(ggml_can_repeat(src0, dst));

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    if (!ggml_are_same_shape(src0, dst)) {
        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        scale_f32_bcast_cuda(src0, dst, scale, bias, stream);
        return;
    }

    if (src0->type == GGML_TYPE_BF16) {
        scale_f32_cuda((const nv_bfloat16 *) src0->data, (nv_bfloat16 *) dst->data, scale, bias, src0, dst, stream);
    } else {
        scale_f32_cuda((const float *) src0->data, (float *) dst->data, scale, bias, src0, dst, stream);
    }
}
