// PMPP_CANARY_04_ca1fa96862 -- held-out canary; MUST NOT appear in any submission
// ============================================================================
// file: paged_kv_decode_reference.cu
// Strong reference pipeline implementation.
// ============================================================================

#include "paged_kv_decode_common.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct PkdReferenceState {
    PkdProblemSpec spec;
    int pages_per_seq;
    int8_t* k_cache;
    int8_t* v_cache;
    float* page_scale;
    int32_t* page_table;
    int32_t* lengths;
    int32_t* next_page;
};

struct PkdWorkspaceLayout {
    int32_t* append_pos;   // B * 2
    int32_t* append_page;  // B * 2
    size_t required_bytes;
};

static size_t pkd_reference_workspace_bytes_for(int B) {
    size_t off = 0;

    off = pkd_align_up_size(off, 128);
    off += sizeof(int32_t) * (size_t)B * (size_t)PKD_MAX_NEW_TOKENS;

    off = pkd_align_up_size(off, 128);
    off += sizeof(int32_t) * (size_t)B * (size_t)PKD_MAX_NEW_TOKENS;

    off = pkd_align_up_size(off, 128);
    return off;
}

static PkdWorkspaceLayout pkd_reference_make_layout(void* workspace, int B) {
    PkdWorkspaceLayout layout{};
    char* base = static_cast<char*>(workspace);
    size_t off = 0;

    off = pkd_align_up_size(off, 128);
    layout.append_pos = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)B * (size_t)PKD_MAX_NEW_TOKENS;

    off = pkd_align_up_size(off, 128);
    layout.append_page = reinterpret_cast<int32_t*>(base + off);
    off += sizeof(int32_t) * (size_t)B * (size_t)PKD_MAX_NEW_TOKENS;

    off = pkd_align_up_size(off, 128);
    layout.required_bytes = off;
    return layout;
}

__device__ __forceinline__ size_t pkd_cache_index_device(
    int page,
    int h,
    int offset,
    int d,
    int Hkv,
    int P,
    int D) {
    return (((size_t)page * (size_t)Hkv + (size_t)h) *
            (size_t)P + (size_t)offset) *
            (size_t)D + (size_t)d;
}

__device__ __forceinline__ float pkd_sanitize_scale_device(float s) {
    return s > 1.0e-6f ? s : 1.0f;
}

__device__ __forceinline__ int8_t pkd_quantize_device(float x, float scale) {
    const float z = x / scale;
    int qi = 0;

    if (z >= 0.0f) {
        qi = static_cast<int>(floorf(z + 0.5f));
    } else {
        qi = static_cast<int>(ceilf(z - 0.5f));
    }

    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;

    return static_cast<int8_t>(qi);
}

__device__ __forceinline__ uint64_t pkd_fnv_byte_device(uint64_t h, uint8_t b) {
    h ^= static_cast<uint64_t>(b);
    h *= 1099511628211ULL;
    return h;
}

__device__ void pkd_fnv_bytes_device(uint64_t* h, const uint8_t* p, size_t n) {
    uint64_t v = *h;
    for (size_t i = 0; i < n; ++i) {
        v = pkd_fnv_byte_device(v, p[i]);
    }
    *h = v;
}

__global__ void pkd_ref_reset_fill_kernel(
    int total_scales,
    float* __restrict__ page_scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_scales) {
        page_scale[i] = 1.0f;
    }
}

static cudaError_t pkd_reference_reset_state(PkdReferenceState* st, cudaStream_t stream) {
    cudaError_t err = cudaSuccess;

    const size_t cache_elems =
        (size_t)st->spec.max_pages *
        (size_t)st->spec.Hkv *
        (size_t)st->spec.page_size *
        (size_t)st->spec.D;

    err = cudaMemsetAsync(st->k_cache, 0, cache_elems * sizeof(int8_t), stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->v_cache, 0, cache_elems * sizeof(int8_t), stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->page_table, 0xFF,
        sizeof(int32_t) * (size_t)st->spec.B * (size_t)st->pages_per_seq,
        stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->lengths, 0,
        sizeof(int32_t) * (size_t)st->spec.B,
        stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(st->next_page, 0, sizeof(int32_t), stream);
    if (err != cudaSuccess) return err;

    const int total_scales = st->spec.max_pages * st->spec.Hkv;
    const int block = 256;
    const int grid = pkd_ceil_div_int(total_scales, block);

    pkd_ref_reset_fill_kernel<<<grid, block, 0, stream>>>(
        total_scales,
        st->page_scale);

    return cudaPeekAtLastError();
}

__global__ void pkd_ref_append_alloc_kernel(
    int B,
    int Hkv,
    int D,
    int P,
    int max_seq_len,
    int max_pages,
    int pages_per_seq,
    int active_count,
    const int32_t* __restrict__ active_seq,
    const int32_t* __restrict__ new_token_count,
    const float* __restrict__ new_scale,
    int32_t* __restrict__ page_table,
    int32_t* __restrict__ lengths,
    int32_t* __restrict__ next_page,
    float* __restrict__ page_scale,
    int32_t* __restrict__ append_pos,
    int32_t* __restrict__ append_page) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (int a = 0; a < B * PKD_MAX_NEW_TOKENS; ++a) {
        append_pos[a] = -1;
        append_page[a] = -1;
    }

    for (int a = 0; a < active_count; ++a) {
        const int seq = active_seq[a];
        if (seq < 0 || seq >= B) {
            continue;
        }

        int ntokens = new_token_count[a];
        if (ntokens < 0) ntokens = 0;
        if (ntokens > PKD_MAX_NEW_TOKENS) ntokens = PKD_MAX_NEW_TOKENS;

        for (int nt = 0; nt < ntokens; ++nt) {
            const int idx = a * PKD_MAX_NEW_TOKENS + nt;
            const int old_len = lengths[seq];

            if (old_len >= max_seq_len) {
                append_pos[idx] = -1;
                append_page[idx] = -1;
                continue;
            }

            const int page_slot = old_len / P;
            const int page_offset = old_len - page_slot * P;

            int page = -1;

            if (page_offset == 0) {
                page = *next_page;
                *next_page = page + 1;

                if (page >= max_pages) {
                    append_pos[idx] = -1;
                    append_page[idx] = -1;
                    continue;
                }

                page_table[seq * pages_per_seq + page_slot] = page;

                for (int h = 0; h < Hkv; ++h) {
                    const size_t scale_idx =
                        ((size_t)a * (size_t)PKD_MAX_NEW_TOKENS + (size_t)nt) *
                        (size_t)Hkv + (size_t)h;
                    page_scale[(size_t)page * (size_t)Hkv + (size_t)h] =
                        pkd_sanitize_scale_device(new_scale[scale_idx]);
                }
            } else {
                page = page_table[seq * pages_per_seq + page_slot];
            }

            append_pos[idx] = old_len;
            append_page[idx] = page;
            lengths[seq] = old_len + 1;
        }
    }
}

__global__ void pkd_ref_quantize_kernel(
    int active_count,
    int Hkv,
    int D,
    int P,
    const float* __restrict__ new_k,
    const float* __restrict__ new_v,
    const int32_t* __restrict__ append_pos,
    const int32_t* __restrict__ append_page,
    const float* __restrict__ page_scale,
    int8_t* __restrict__ k_cache,
    int8_t* __restrict__ v_cache) {
    const int total = active_count * PKD_MAX_NEW_TOKENS * Hkv * D;
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) return;

    int tmp = linear;
    const int d = tmp % D;
    tmp /= D;
    const int h = tmp % Hkv;
    tmp /= Hkv;
    const int nt = tmp % PKD_MAX_NEW_TOKENS;
    const int a = tmp / PKD_MAX_NEW_TOKENS;

    const int idx = a * PKD_MAX_NEW_TOKENS + nt;
    const int pos = append_pos[idx];
    const int page = append_page[idx];

    if (pos < 0 || page < 0) {
        return;
    }

    const int offset = pos % P;
    const float scale = page_scale[(size_t)page * (size_t)Hkv + (size_t)h];

    const size_t input_idx =
        ((size_t)a * (size_t)PKD_MAX_NEW_TOKENS + (size_t)nt) *
        (size_t)Hkv * (size_t)D +
        (size_t)h * (size_t)D +
        (size_t)d;

    const size_t cache_idx = pkd_cache_index_device(page, h, offset, d, Hkv, P, D);

    k_cache[cache_idx] = pkd_quantize_device(new_k[input_idx], scale);
    v_cache[cache_idx] = pkd_quantize_device(new_v[input_idx], scale);
}

__device__ __forceinline__ float pkd_warp_sum(float v){
    for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffffu,v,o);
    return v;
}
__global__ void pkd_ref_attention_kernel(
    int B,int Hq,int Hkv,int D,int P,int max_seq_len,int pages_per_seq,
    int active_count,int window_size,
    const int32_t* __restrict__ active_seq,const float* __restrict__ q,
    const int32_t* __restrict__ page_table,const int32_t* __restrict__ lengths,
    const int8_t* __restrict__ k_cache,const int8_t* __restrict__ v_cache,
    const float* __restrict__ page_scale,float* __restrict__ y){
    __shared__ float s_red[4];
    const int a=blockIdx.x, hq=blockIdx.y, t=threadIdx.x;
    if(a>=active_count||hq>=Hq) return;
    const int seq=active_seq[a];
    const int group=Hq/Hkv; const int kvh=hq/group;
    const size_t yo=((size_t)a*(size_t)Hq+(size_t)hq)*(size_t)D+(size_t)t;
    if(seq<0||seq>=B){ y[yo]=0.0f; return; }
    const int len=lengths[seq];
    const int W=window_size>max_seq_len?max_seq_len:window_size;
    int start=len-W; if(start<0) start=0;
    if(len<=start){ y[yo]=0.0f; return; }
    const int lane=t&31, warp=t>>5, nwarps=blockDim.x>>5;
    const float inv_sqrt_d=rsqrtf((float)D);
    const float qv=q[((size_t)a*(size_t)Hq+(size_t)hq)*(size_t)D+(size_t)t];
    float acc=0.0f, m=-3.4028234663852886e38f, l=0.0f;
    for(int pos=start;pos<len;++pos){
        const int page_slot=pos/P; const int offset=pos-page_slot*P;
        const int page=page_table[seq*pages_per_seq+page_slot];
        const float scale=page_scale[(size_t)page*(size_t)Hkv+(size_t)kvh];
        const size_t ci=(((size_t)page*(size_t)Hkv+(size_t)kvh)*(size_t)P+(size_t)offset)*(size_t)D+(size_t)t;
        const float kv=(float)k_cache[ci]*scale;
        float dot=pkd_warp_sum(qv*kv);
        if(lane==0) s_red[warp]=dot;
        __syncthreads();
        dot=0.0f;
        for(int w=0;w<nwarps;++w) dot+=s_red[w];
        __syncthreads();
        const float logit=dot*inv_sqrt_d;
        const float nm=fmaxf(m,logit);
        const float corr=__expf(m-nm);
        const float ww=__expf(logit-nm);
        const float vv=(float)v_cache[ci]*scale;
        acc=acc*corr+ww*vv; l=l*corr+ww; m=nm;
    }
    y[yo]=l>0.0f?acc/l:0.0f;
}

__global__ void pkd_ref_copy_lengths_kernel(
    int B,
    const int32_t* __restrict__ lengths,
    int32_t* __restrict__ out_lengths) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B) {
        out_lengths[i] = lengths[i];
    }
}

__global__ void pkd_ref_checksum_kernel(
    int B,
    int Hkv,
    int D,
    int P,
    int max_pages,
    int pages_per_seq,
    const int32_t* __restrict__ lengths,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ next_page,
    const float* __restrict__ page_scale,
    const int8_t* __restrict__ k_cache,
    const int8_t* __restrict__ v_cache,
    uint64_t* __restrict__ checksum) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint64_t h = 1469598103934665603ULL;

    pkd_fnv_bytes_device(
        &h,
        reinterpret_cast<const uint8_t*>(next_page),
        sizeof(int32_t));

    pkd_fnv_bytes_device(
        &h,
        reinterpret_cast<const uint8_t*>(lengths),
        sizeof(int32_t) * (size_t)B);

    pkd_fnv_bytes_device(
        &h,
        reinterpret_cast<const uint8_t*>(page_table),
        sizeof(int32_t) * (size_t)B * (size_t)pages_per_seq);

    int used_pages = *next_page;
    if (used_pages < 0) used_pages = 0;
    if (used_pages > max_pages) used_pages = max_pages;

    const size_t page_cache_elems =
        (size_t)Hkv * (size_t)P * (size_t)D;

    for (int p = 0; p < used_pages; ++p) {
        pkd_fnv_bytes_device(
            &h,
            reinterpret_cast<const uint8_t*>(&page_scale[(size_t)p * (size_t)Hkv]),
            sizeof(float) * (size_t)Hkv);

        pkd_fnv_bytes_device(
            &h,
            reinterpret_cast<const uint8_t*>(&k_cache[(size_t)p * page_cache_elems]),
            sizeof(int8_t) * page_cache_elems);

        pkd_fnv_bytes_device(
            &h,
            reinterpret_cast<const uint8_t*>(&v_cache[(size_t)p * page_cache_elems]),
            sizeof(int8_t) * page_cache_elems);
    }

    checksum[0] = h;
}

extern "C" size_t solution_workspace_bytes(const PkdProblemSpec* spec) {
    if (!pkd_validate_problem_spec(spec)) return 0;
    return pkd_reference_workspace_bytes_for(spec->B);
}

extern "C" cudaError_t solution_init(
    const PkdProblemSpec* spec,
    void** state_out,
    cudaStream_t stream) {
    if (!pkd_validate_problem_spec(spec) || !state_out) {
        return cudaErrorInvalidValue;
    }

    PkdReferenceState* st = static_cast<PkdReferenceState*>(malloc(sizeof(PkdReferenceState)));
    if (!st) {
        return cudaErrorMemoryAllocation;
    }

    memset(st, 0, sizeof(PkdReferenceState));
    memcpy(&st->spec, spec, sizeof(PkdProblemSpec));
    st->pages_per_seq = pkd_pages_per_seq(spec->max_seq_len, spec->page_size);

    const size_t cache_elems =
        (size_t)spec->max_pages *
        (size_t)spec->Hkv *
        (size_t)spec->page_size *
        (size_t)spec->D;

    cudaError_t err = cudaSuccess;

    err = cudaMalloc(reinterpret_cast<void**>(&st->k_cache), cache_elems * sizeof(int8_t));
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->v_cache), cache_elems * sizeof(int8_t));
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->page_scale),
        sizeof(float) * (size_t)spec->max_pages * (size_t)spec->Hkv);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->page_table),
        sizeof(int32_t) * (size_t)spec->B * (size_t)st->pages_per_seq);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->lengths),
        sizeof(int32_t) * (size_t)spec->B);
    if (err != cudaSuccess) goto fail;

    err = cudaMalloc(reinterpret_cast<void**>(&st->next_page), sizeof(int32_t));
    if (err != cudaSuccess) goto fail;

    err = pkd_reference_reset_state(st, stream);
    if (err != cudaSuccess) goto fail;

    *state_out = st;
    return cudaSuccess;

fail:
    if (st->k_cache) cudaFree(st->k_cache);
    if (st->v_cache) cudaFree(st->v_cache);
    if (st->page_scale) cudaFree(st->page_scale);
    if (st->page_table) cudaFree(st->page_table);
    if (st->lengths) cudaFree(st->lengths);
    if (st->next_page) cudaFree(st->next_page);
    free(st);
    return err;
}

extern "C" cudaError_t solution_run(
    void* state,
    const PkdRunSpec* run,
    const void* inputs_void,
    void* outputs_void,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
    if (!state || !inputs_void || !outputs_void || !workspace) {
        return cudaErrorInvalidValue;
    }

    PkdReferenceState* st = static_cast<PkdReferenceState*>(state);

    if (!pkd_validate_run_spec(run, &st->spec)) {
        return cudaErrorInvalidValue;
    }

    const PkdInputs* in = static_cast<const PkdInputs*>(inputs_void);
    PkdOutputs* out = static_cast<PkdOutputs*>(outputs_void);

    if (!in->active_seq || !in->new_token_count || !in->new_k ||
        !in->new_v || !in->new_scale || !in->q ||
        !out->y || !out->lengths || !out->state_checksum) {
        return cudaErrorInvalidValue;
    }

    PkdWorkspaceLayout layout = pkd_reference_make_layout(workspace, st->spec.B);
    if (workspace_bytes < layout.required_bytes) {
        return cudaErrorInvalidValue;
    }

    const int B = st->spec.B;
    const int Hq = st->spec.Hq;
    const int Hkv = st->spec.Hkv;
    const int D = st->spec.D;
    const int P = st->spec.page_size;
    const int max_seq_len = st->spec.max_seq_len;
    const int max_pages = st->spec.max_pages;
    const int active_count = run->active_count;

    cudaError_t err = cudaSuccess;

    pkd_ref_append_alloc_kernel<<<1, 1, 0, stream>>>(
        B,
        Hkv,
        D,
        P,
        max_seq_len,
        max_pages,
        st->pages_per_seq,
        active_count,
        in->active_seq,
        in->new_token_count,
        in->new_scale,
        st->page_table,
        st->lengths,
        st->next_page,
        st->page_scale,
        layout.append_pos,
        layout.append_page);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    const int q_total = active_count * PKD_MAX_NEW_TOKENS * Hkv * D;
    const int q_block = 256;
    const int q_grid = pkd_ceil_div_int(q_total, q_block);

    if (q_total > 0) {
        pkd_ref_quantize_kernel<<<q_grid, q_block, 0, stream>>>(
            active_count,
            Hkv,
            D,
            P,
            in->new_k,
            in->new_v,
            layout.append_pos,
            layout.append_page,
            st->page_scale,
            st->k_cache,
            st->v_cache);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    if (active_count > 0) {
        dim3 attn_grid(active_count, Hq, 1);
        pkd_ref_attention_kernel<<<attn_grid, D, 0, stream>>>(
            B,
            Hq,
            Hkv,
            D,
            P,
            max_seq_len,
            st->pages_per_seq,
            active_count,
            run->window_size,
            in->active_seq,
            in->q,
            st->page_table,
            st->lengths,
            st->k_cache,
            st->v_cache,
            st->page_scale,
            out->y);
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) return err;
    }

    const int len_block = 256;
    const int len_grid = pkd_ceil_div_int(B, len_block);

    pkd_ref_copy_lengths_kernel<<<len_grid, len_block, 0, stream>>>(
        B,
        st->lengths,
        out->lengths);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    pkd_ref_checksum_kernel<<<1, 1, 0, stream>>>(
        B,
        Hkv,
        D,
        P,
        max_pages,
        st->pages_per_seq,
        st->lengths,
        st->page_table,
        st->next_page,
        st->page_scale,
        st->k_cache,
        st->v_cache,
        out->state_checksum);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

extern "C" cudaError_t solution_reset(
    void* state,
    cudaStream_t stream) {
    if (!state) return cudaErrorInvalidValue;
    return pkd_reference_reset_state(static_cast<PkdReferenceState*>(state), stream);
}

extern "C" void solution_destroy(void* state) {
    if (!state) return;

    PkdReferenceState* st = static_cast<PkdReferenceState*>(state);
    if (st->k_cache) cudaFree(st->k_cache);
    if (st->v_cache) cudaFree(st->v_cache);
    if (st->page_scale) cudaFree(st->page_scale);
    if (st->page_table) cudaFree(st->page_table);
    if (st->lengths) cudaFree(st->lengths);
    if (st->next_page) cudaFree(st->next_page);
    free(st);
}
