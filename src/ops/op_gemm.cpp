// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/net2_impl.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv { namespace dnn {

// The implementation is derived (but then heavily rewritten)
// from BLISLAB code (https://github.com/flame/blislab).
// Below is the original copyright and the license

/*
 * --------------------------------------------------------------------------
 * BLISLAB
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * bl_sgemm.c
 * */

#if defined __ARM_NEON
#define _CV_SIMD_NEON 1
#endif

#if defined __avx2__
#define _CV_SIMD_AVX2 1
#endif

#define _CV_GEMM_STORAGE (1<<20)
#define _CV_GEMM_MAX_STACKBUF (1 << 14)

#define _CV_SGEMM_MC 64
#define _CV_SGEMM_NC 240
#define _CV_SGEMM_VOL (1<<18)
#define _CV_SGEMM_MR 8
#define _CV_SGEMM_NR 12

#define _CV_GEMM_IMPLEMENT_PACK(N, suffix, styp, dtyp) \
static void gemm_pack##N##suffix( int64_t m, int64_t k, const void* A_, \
                                  int64_t lda0, int64_t lda1, void* packA_ ) \
{ \
    const styp* A = (const styp*)A_; \
    dtyp* packA = (dtyp*)packA_; \
    for( int64_t i = 0; i < m; i += N ) { \
        if (i + N-1 < m) { \
            const styp* a_ptr = A + lda0*i; \
            for( int64_t j = 0; j < k*lda1; packA += N, j += lda1 ) \
            { \
                _CV_GEMM_LOAD_TO_BUF_##N(styp); \
                _CV_GEMM_PACK##suffix##_##N(buf, packA); \
            } \
        } else { \
            const styp* a_ptr[N]; \
            for (int k = 0; k < N; k++) a_ptr[k] = A + lda0*(i+k < m ? i+k : i); \
            for( int64_t j = 0; j < k*lda1; packA += N, j += lda1 ) \
            { \
                _CV_GEMM_LOAD_TO_BUF_BORDERS_##N(styp); \
                _CV_GEMM_PACK##suffix##_##N(buf, packA); \
            } \
        } \
    } \
}

#define _CV_GEMM_LOAD_TO_BUF_8(styp) \
    styp buf[] = { \
        a_ptr[j], a_ptr[j+lda0], a_ptr[j+lda0*2], a_ptr[j+lda0*3], \
        a_ptr[j+lda0*4], a_ptr[j+lda0*5], a_ptr[j+lda0*6], a_ptr[j+lda0*7] }

#define _CV_GEMM_LOAD_TO_BUF_BORDERS_8(styp) \
    styp buf[] = { \
        a_ptr[0][j], a_ptr[1][j], a_ptr[2][j], a_ptr[3][j], \
        a_ptr[4][j], a_ptr[5][j], a_ptr[6][j], a_ptr[7][j] }

#define _CV_GEMM_LOAD_TO_BUF_12(styp) \
    styp buf[] = { \
        a_ptr[j], a_ptr[j+lda0], a_ptr[j+lda0*2], a_ptr[j+lda0*3], \
        a_ptr[j+lda0*4], a_ptr[j+lda0*5], a_ptr[j+lda0*6], a_ptr[j+lda0*7], \
        a_ptr[j+lda0*8], a_ptr[j+lda0*9], a_ptr[j+lda0*10], a_ptr[j+lda0*11] }

#define _CV_GEMM_LOAD_TO_BUF_BORDERS_12(styp) \
    styp buf[] = { \
        a_ptr[0][j], a_ptr[1][j], a_ptr[2][j], a_ptr[3][j], \
        a_ptr[4][j], a_ptr[5][j], a_ptr[6][j], a_ptr[7][j], \
        a_ptr[8][j], a_ptr[9][j], a_ptr[10][j], a_ptr[11][j] }

#define _CV_GEMM_PACK_COPY(src, dst, N) \
    memcpy((dst), (src), N*sizeof(src[0]))
#define _CV_GEMM_PACK_f32_8(src, dst) _CV_GEMM_PACK_COPY((src), (dst), 8)
#define _CV_GEMM_PACK_f32_12(src, dst) _CV_GEMM_PACK_COPY((src), (dst), 12)

#if _CV_SIMD_NEON

#define _CV_GEMM_PACK_f16f32_8(src, dst) \
    float16x8_t x0 = vld1q_f16((const __fp16*)(src)); \
    float32x4_t y0 = vcvt_f32_f16(vget_low_f16(x0)); \
    float32x4_t y1 = vcvt_f32_f16(vget_high_f16(x0)); \
    vst1q_f32((dst), y0); vst1q_f32((dst) + 4, y1)

#define _CV_GEMM_PACK_f16f32_12(src, dst) \
    float16x8_t x0 = vld1q_f16((const __fp16*)(src)); \
    float16x4_t x1 = vld1_f16((const __fp16*)(src) + 8); \
    float32x4_t y0 = vcvt_f32_f16(vget_low_f16(x0)); \
    float32x4_t y1 = vcvt_f32_f16(vget_high_f16(x0)); \
    float32x4_t y2 = vcvt_f32_f16(x1); \
    vst1q_f32((dst), y0); \
    vst1q_f32((dst) + 4, y1); \
    vst1q_f32((dst) + 8, y2)

#define _CV_GEMM_PACK_f32f16_8(src, dst) \
    float32x4_t x0 = vld1q_f32((src)); \
    float32x4_t x1 = vld1q_f32((src) + 4); \
    float16x8_t y0 = vcombine_f16(vcvt_f16_f32(x0), vcvt_f16_f32(x1)); \
    vst1q_f16((__fp16*)(dst), y0)

#elif _CV_SIMD_AVX2

#define _CV_GEMM_PACK_f16f32_8(src, dst) \
    __m128i x0 = _mm_loadu_si128((const __m128i*)(src)); \
    __m128 y0 = _mm_cvtph_ps(x0); \
    __m128 y1 = _mm_cvtph_ps(_mm_unpackhi_epi64(x0, x0)); \
    _mm_storeu_ps((dst), y0); \
    _mm_storeu_ps((dst) + 4, y1)

#define _CV_GEMM_PACK_f16f32_12(src, dst) \
    __m128i x0 = _mm_loadu_si128((const __m128i*)(src)); \
    __m128i x1 = _mm_loadl_epi64((const __m128i*)((src) + 8)); \
    __m128 y0 = _mm_cvtph_ps(x0); \
    __m128 y1 = _mm_cvtph_ps(_mm_unpackhi_epi64(x0, x0)); \
    __m128 y2 = _mm_cvtph_ps(x1); \
    _mm_storeu_ps((dst), y0); \
    _mm_storeu_ps((dst) + 4, y1); \
    _mm_storeu_ps((dst) + 8, y2)

#define _CV_GEMM_PACK_f32f16_8(src, dst) \
    __m128i x0 = _mm_loadu_ps((src)); \
    __m128i x1 = _mm_loadu_ps((src) + 4); \
    __m128i y0 = _mm_cvtps_ph(x0, 0); \
    __m128i y1 = _mm_cvtps_ph(x1, 0); \
    _mm_storeu_si128((__m128i*)(dst), _mm_unpacklo_epi64(y0, y1))

#else

#define _CV_GEMM_PACK_TO_FLOAT(src, dst, N) \
    for (int k = 0; k < N; k++) (dst)[k] = float((src)[k])

#define _CV_GEMM_PACK_TO_FLOAT16(src, dst, N) \
    for (int k = 0; k < N; k++) (dst)[k] = hfloat((src)[k])

#define _CV_GEMM_PACK_f16f32_8(src, dst) _CV_GEMM_PACK_TO_FLOAT((src), (dst), 8)
#define _CV_GEMM_PACK_f16f32_12(src, dst) _CV_GEMM_PACK_TO_FLOAT((src), (dst), 12)
#define _CV_GEMM_PACK_f32f16_8(src, dst) _CV_GEMM_PACK_TO_FLOAT16((src), (dst), 8)

#endif

_CV_GEMM_IMPLEMENT_PACK(8, _f32, float, float)
_CV_GEMM_IMPLEMENT_PACK(12, _f32, float, float)

_CV_GEMM_IMPLEMENT_PACK(8, _f16f32, hfloat, float)
_CV_GEMM_IMPLEMENT_PACK(12, _f16f32, hfloat, float)

typedef void (*gemm_packer_t)(int64_t m, int64_t k, const void* A_,
                              int64_t lda0, int64_t lda1, void* packA_);

static void gemm8x12_f32(int64_t k, const char *a_, const char *b_,
                         float *c_, int64_t ldc, float alpha)
{
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* c = c_;

#if _CV_SIMD_NEON
    float32x4_t s00 = vdupq_n_f32(0.f), s01 = s00, s02 = s00;
    float32x4_t s10 = s00, s11 = s00, s12 = s00;
    float32x4_t s20 = s00, s21 = s00, s22 = s00;
    float32x4_t s30 = s00, s31 = s00, s32 = s00;
    float32x4_t s40 = s00, s41 = s00, s42 = s00;
    float32x4_t s50 = s00, s51 = s00, s52 = s00;
    float32x4_t s60 = s00, s61 = s00, s62 = s00;
    float32x4_t s70 = s00, s71 = s00, s72 = s00;

    for( int64_t p = 0; p < k; p++, a += _CV_SGEMM_MR, b += _CV_SGEMM_NR )
    {
        float32x4_t a0 = vld1q_f32(a);
        float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8);

        s00 = vfmaq_laneq_f32(s00, b0, a0, 0);
        s01 = vfmaq_laneq_f32(s01, b1, a0, 0);
        s02 = vfmaq_laneq_f32(s02, b2, a0, 0);
        s10 = vfmaq_laneq_f32(s10, b0, a0, 1);
        s11 = vfmaq_laneq_f32(s11, b1, a0, 1);
        s12 = vfmaq_laneq_f32(s12, b2, a0, 1);

        s20 = vfmaq_laneq_f32(s20, b0, a0, 2);
        s21 = vfmaq_laneq_f32(s21, b1, a0, 2);
        s22 = vfmaq_laneq_f32(s22, b2, a0, 2);
        s30 = vfmaq_laneq_f32(s30, b0, a0, 3);
        s31 = vfmaq_laneq_f32(s31, b1, a0, 3);
        s32 = vfmaq_laneq_f32(s32, b2, a0, 3);

        a0 = vld1q_f32(a + 4);

        s40 = vfmaq_laneq_f32(s40, b0, a0, 0);
        s41 = vfmaq_laneq_f32(s41, b1, a0, 0);
        s42 = vfmaq_laneq_f32(s42, b2, a0, 0);
        s50 = vfmaq_laneq_f32(s50, b0, a0, 1);
        s51 = vfmaq_laneq_f32(s51, b1, a0, 1);
        s52 = vfmaq_laneq_f32(s52, b2, a0, 1);

        s60 = vfmaq_laneq_f32(s60, b0, a0, 2);
        s61 = vfmaq_laneq_f32(s61, b1, a0, 2);
        s62 = vfmaq_laneq_f32(s62, b2, a0, 2);
        s70 = vfmaq_laneq_f32(s70, b0, a0, 3);
        s71 = vfmaq_laneq_f32(s71, b1, a0, 3);
        s72 = vfmaq_laneq_f32(s72, b2, a0, 3);
    }

    float32x4_t c0, c1, c2, c3, c4, c5, valpha = vdupq_n_f32(alpha);
    #define _CV_SGEMM_FINIT(row0, row1) \
        c0 = vld1q_f32(c + row0*ldc); \
        c1 = vld1q_f32(c + row0*ldc + 4); \
        c2 = vld1q_f32(c + row0*ldc + 8); \
        c3 = vld1q_f32(c + row1*ldc); \
        c4 = vld1q_f32(c + row1*ldc + 4); \
        c5 = vld1q_f32(c + row1*ldc + 8); \
        c0 = vfmaq_f32(c0, s##row0##0, valpha); \
        c1 = vfmaq_f32(c1, s##row0##1, valpha); \
        c2 = vfmaq_f32(c2, s##row0##2, valpha); \
        c3 = vfmaq_f32(c3, s##row1##0, valpha); \
        c4 = vfmaq_f32(c4, s##row1##1, valpha); \
        c5 = vfmaq_f32(c5, s##row1##2, valpha); \
        vst1q_f32(c + row0*ldc, c0); \
        vst1q_f32(c + row0*ldc + 4, c1); \
        vst1q_f32(c + row0*ldc + 8, c2); \
        vst1q_f32(c + row1*ldc, c3); \
        vst1q_f32(c + row1*ldc + 4, c4); \
        vst1q_f32(c + row1*ldc + 8, c5)

    _CV_SGEMM_FINIT(0, 1);
    _CV_SGEMM_FINIT(2, 3);
    _CV_SGEMM_FINIT(4, 5);
    _CV_SGEMM_FINIT(6, 7);
#else
    // we give a chance to compiler to vectorize the loop by storing
    // intermediate sums in local buffer with compile-time-constant size.
    float sbuf[_CV_SGEMM_MR*_CV_SGEMM_NR];
    memset(sbuf, 0, sizeof(sbuf));
    for( int64_t p = 0; p < k; p++ ) {
        for( int i = 0; i < _CV_SGEMM_MR; i++ ) {
            float ai = a[_CV_SGEMM_MR*p + i];
            for( int j = 0; j < _CV_SGEMM_NR; j++ )
                sbuf[i*_CV_SGEMM_NR+j] += b[_CV_SGEMM_NR*p + j]*ai;
        }
    }
    for (int i = 0; i < _CV_SGEMM_MR; i++) {
        for (int j = 0; j < _CV_SGEMM_NR; j++)
            c[i*ldc + j] += alpha*sbuf[i*_CV_SGEMM_NR+j];
    }
#endif
}

static void gemm_thin(float alpha, float beta, int64_t M, int64_t N, int64_t K,
                      int a_typ, const char* a_, int64_t lda0, int64_t lda1,
                      int b_typ, const char* b_, int64_t ldb,
                      int64_t mbias, int bias_typ, void* bias_, int64_t ldbias0,
                      int c_typ, char* c_, int64_t ldc, int num_threads)
{
    int64_t nsubtasks = 1, Nblocks = 1;
    if (num_threads > 1 && (uint64_t)M*N*K >= 100000) {
        if (M < num_threads)
            Nblocks = num_threads/M;
    } else {
        num_threads = 1;
    }
    nsubtasks = M*Nblocks;
    if (a_typ == CV_32F && b_typ == CV_32F && c_typ == CV_32F && bias_typ == CV_32F) {
        const float* a = (const float*)a_;
        parallel_for_(Range(0, (int)nsubtasks), [&](const Range& r) {
            int64_t tile0 = r.start, tile1 = r.end;
            for( ; tile0 < tile1; tile0++ ) {
                int64_t i = tile0/Nblocks;
                int64_t nb = tile0 - i*Nblocks;
                int64_t j0 = nb*N/Nblocks, j1 = (nb+1)*N/Nblocks;
                int64_t j, k;
                float* c_i = (float*)c_ + i*ldc;
                for( j = j0; j < j1; j++ ) c_i[j] = 0.f;
                for( k = 0; k < K; k++ ) {
                    const float* b_k = (const float*)b_ + k*ldb;
                    float aval = alpha*a[i*lda0 + k*lda1];
                    for( j = j0; j < j1; j++ )
                        c_i[j] += aval*b_k[j];
                }
                if (bias_) {
                    const float* bias_i = (const float*)bias_ + i*ldbias0;
                    for( j = j0; j < j1; j++ ) c_i[j] += bias_i[j]*beta;
                }
            }
        }, num_threads);
    } else {
        if ((a_typ != CV_32F && a_typ != CV_16F) ||
            (b_typ != CV_32F && b_typ != CV_16F) ||
            bias_typ != CV_32F ||
            c_typ != CV_32F) {
            CV_Error(Error::StsNotImplemented, "");
        }
        parallel_for_(Range(0, (int)nsubtasks), [&](const Range& r) {
            int64_t tile0 = r.start, tile1 = r.end;
            for( ; tile0 < tile1; tile0++ ) {
                int64_t i = tile0/Nblocks;
                int64_t nb = tile0 - i*Nblocks;
                int64_t j0 = nb*N/Nblocks, j1 = (nb+1)*N/Nblocks, Nt = j1 - j0;
                int64_t j, k;

                float* c_i = (float*)c_ + i*ldc + j0;
                for( j = 0; j < Nt; j++ ) c_i[j] = 0.f;
                for( k = 0; k < K; k++ ) {
                    float aval = alpha*(a_typ == CV_32F ? *((const float*)a_ + i*lda0 + k*lda1) :
                                        float(*((const hfloat*)a_ + i*lda0 + k*lda1)));
                    if (b_typ == CV_32F) {
                        const float* b_k = (const float*)b_ + k*ldb + j0;
                        for( j = 0; j < Nt; j++ )
                            c_i[j] += aval*b_k[j];
                    } else {
                        const hfloat* b_k = (const hfloat*)b_ + k*ldb + j0;
                        j = 0;
                    #if _CV_SIMD_NEON
                        float32x4_t va = vdupq_n_f32(aval);
                        for (; j <= Nt - 8; j += 8) {
                            float16x8_t bval = vld1q_f16((const __fp16*)(b_k + j));
                            float32x4_t b0 = vcvt_f32_f16(vget_low_f16(bval));
                            float32x4_t b1 = vcvt_f32_f16(vget_high_f16(bval));
                            float32x4_t c0 = vld1q_f32(c_i + j);
                            float32x4_t c1 = vld1q_f32(c_i + j + 4);
                            c0 = vfmaq_f32(c0, b0, va);
                            c1 = vfmaq_f32(c1, b1, va);
                            vst1q_f32(c_i + j, c0);
                            vst1q_f32(c_i + j + 4, c1);
                        }
                    #elif _CV_SIMD_AVX2
                        __m128 va = _mm_set1_ps(aval);
                        for (; j <= Nt - 8; j += 8) {
                            __m128i bval = _mm_loadu_si128((const __m128i*)(b_k + j));
                            __m128 b0 = _mm_cvtph_ps(bval);
                            __m128 b1 = _mm_cvtph_ps(_mm_unpackhi_epi64(bval, bval));
                            __m128 c0 = _mm_loadu_ps(c_i + j);
                            __m128 c1 = _mm_loadu_ps(c_i + j + 4);
                            c0 = _mm_fmadd_ps(b0, va, c0);
                            c1 = _mm_fmadd_ps(b1, va, c1);
                            _mm_storeu_ps(c_i + j, c0);
                            _mm_storeu_ps(c_i + j + 4, c1);
                        }
                    #endif
                        for (; j < Nt; j++) {
                            float bval = float(b_k[j]);
                            c_i[j] += aval*bval;
                        }
                    }
                }
                if (bias_) {
                    const float* bias_i = (const float*)bias_ + i*ldbias0 + j0;
                    for (j = 0; j < Nt; j++) {
                        c_i[j] += bias_i[j]*beta;
                    }
                }
            }
        }, num_threads);
    }
}

static void gemm_macro_kernel( int typ, int64_t m, int64_t n, int64_t k,
                               const char *packA, const char *packB,
                               float alpha, float *c, int64_t ldc0,
                               int64_t MR, int64_t NR )
{
    int64_t esz = CV_ELEM_SIZE1(typ);
    assert(typ == CV_32F || typ == CV_64F);

    float tempC[_CV_SGEMM_MR*_CV_SGEMM_NR]; // make sure the buffer is big enough
    for( int64_t i = 0; i < m; i += MR ) {
        for( int64_t j = 0; j < n; j += NR ) {
            float* cptr0 = &c[i * ldc0 + j];
            float* cptr = cptr0;
            int64_t ldc = ldc0;
            int64_t mr = m - i < MR ? m - i : MR;
            int64_t nr = n - j < NR ? n - j : NR;
            size_t c_nr_esz = nr*sizeof(c[0]);
            bool partial = (bool)((mr < MR) | (nr < NR));
            if (partial) {
                memset(tempC, 0, sizeof(tempC));
                cptr = tempC;
                ldc = NR;
                for(int64_t p = 0; p < mr; p++)
                    memcpy(cptr + p*ldc, cptr0 + p*ldc0, c_nr_esz);
            }

            gemm8x12_f32(k, packA + i * k * esz, packB + j * k * esz, cptr, ldc, alpha);
            if (partial) {
                for(int64_t p = 0; p < mr; p++)
                    memcpy(cptr0 + p*ldc0, cptr + p*ldc, c_nr_esz);
            }
        }
    }
}

static void gemm( bool tA, bool tB, float alpha, float beta,
                  int64_t ma, int64_t na, int a_typ, const void *a_, int64_t lda0, int64_t lda1,
                  int64_t mb, int64_t nb, int b_typ, const void *b_, int64_t ldb0, int64_t ldb1,
                  int64_t mbias, int64_t nbias, int bias_typ, void* bias_, int64_t ldbias0,
                  int c_typ, void *c_, int64_t ldc, int num_threads )
{
    const char* a = (const char*)a_;
    const char* b = (const char*)b_;
    char* c = (char*)c_;

    num_threads = num_threads <= 0 ? 1 : num_threads;
    int64_t t, M = tA ? na : ma, N = tB ? mb : nb, K = tA ? ma : na;

    if (tA) std::swap(lda0, lda1);
    if (tB) std::swap(ldb0, ldb1);

    if (!(ma > 0 && na > 0 && mb > 0 && nb > 0 && a && b && c)) {
        CV_Error(Error::StsNotImplemented, "");
    }

    if (beta == 0.f)
        bias_ = nullptr;
    if (!bias_) {
        mbias = 1;
        bias_typ = CV_32F;
    }
    if (mbias == 1)
        ldbias0 = 0;

    CV_Assert(mbias == 1 || mbias == M);
    CV_Assert(nbias == N);
    CV_Assert(bias_typ == CV_32F);
    CV_Assert(c_typ == CV_32F);

    //printf("gemm: a_typ=%d, b_typ=%d, c_typ=%d, tA=%d, tB=%d, alpha=%.3f, beta=%.3f\n",
    //       a_typ, b_typ, c_typ, (int)tA, (int)tB, alpha, beta);

    if (!tB && ldb1 == 1 && (M <= 4 || (uint64_t)M*N*K <= 10000)) {
        gemm_thin(alpha, beta, M, N, K, a_typ, a, lda0, lda1,
                  b_typ, b, ldb0, mbias, bias_typ, bias_, ldbias0,
                  c_typ, c, ldc, num_threads);
        return;
    }

    {
        gemm_packer_t a_packer, b_packer;
        int w_typ = c_typ != CV_16F ? c_typ : CV_32F;
        int64_t a_esz = CV_ELEM_SIZE1(a_typ), b_esz = CV_ELEM_SIZE1(b_typ);
        int64_t c_esz = CV_ELEM_SIZE1(c_typ), w_esz = CV_ELEM_SIZE1(w_typ);
        int64_t GEMM_MC = _CV_SGEMM_MC, GEMM_NC = _CV_SGEMM_NC;
        int64_t GEMM_MR = _CV_SGEMM_MR, GEMM_NR = _CV_SGEMM_NR;
        int64_t GEMM_VOL = _CV_SGEMM_VOL;

        int64_t MC = (((GEMM_MC < M ? GEMM_MC : M) + GEMM_MR-1) / GEMM_MR) * GEMM_MR;
        int64_t NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR-1) / GEMM_NR) * GEMM_NR;
        int64_t KC = _CV_GEMM_STORAGE/((MC+NC)*w_esz);
        KC = KC > 8 ? KC : 8;
        KC = KC < K ? KC : K;

        size_t bufsize = KC*(MC+NC)*w_esz;
        int64_t m_tiles = (M + MC - 1)/MC;
        int64_t n_tiles = (N + NC - 1)/NC;
        int64_t total_tiles = m_tiles * n_tiles;
        int64_t ntasks = (int64_t)num_threads < total_tiles ?
        (int64_t)num_threads : total_tiles;
        if ((uint64_t)total_tiles*K < 10000)
            ntasks = 1;

        a_packer =
        a_typ == CV_32F ? gemm_pack8_f32 :
        a_typ == CV_16F ? gemm_pack8_f16f32 : 0;

        b_packer =
        b_typ == CV_32F ? gemm_pack12_f32 :
        b_typ == CV_16F ? gemm_pack12_f16f32 : 0;

        if (!a_packer || !b_packer) {
            CV_Error(Error::StsNotImplemented, "");
        }

        parallel_for_(Range(0, (int)ntasks), [&](const Range& r) {
            AutoBuffer<char> packAbuf(bufsize);
            char* packA = packAbuf.data();
            for( int tid = r.start; tid < r.end; tid++ )
            {
                char* packB = packA + KC*MC*w_esz;
                int64_t start_tile = total_tiles*tid/ntasks;
                int64_t end_tile = total_tiles*(tid+1)/ntasks;

                for( int64_t tile_idx = start_tile; tile_idx < end_tile; tile_idx++ )
                {
                    int64_t i0 = (tile_idx / n_tiles)*MC;
                    int64_t j0 = (tile_idx % n_tiles)*NC;
                    int64_t mc = M - i0 < MC ? M - i0 : MC;
                    int64_t nc = N - j0 < NC ? N - j0 : NC;
                    int64_t ldc_block = ldc;
                    float* c_block = (float*)c + i0 * ldc + j0;

                    for(int64_t i = 0; i < mc; i++) {
                        for (int64_t j = 0; j < nc; j++)
                            c_block[i*ldc_block + j] = 0.f;
                    }

                    for( int64_t k0 = 0; k0 < K; k0 += KC )
                    {
                        int64_t kc = K - k0 < KC ? K - k0 : KC;
                        a_packer(mc, kc, a + (i0*lda0 + k0*lda1)*a_esz, lda0, lda1, packA);
                        b_packer(nc, kc, b + (k0*ldb0 + j0*ldb1)*b_esz, ldb1, ldb0, packB);
                        gemm_macro_kernel(w_typ, mc, nc, kc, packA, packB, alpha,
                                          c_block, ldc_block, GEMM_MR, GEMM_NR);
                    }

                    if (bias_) {
                        for(int64_t i = 0; i < mc; i++) {
                            float* c_i = (float*)c_block + i*ldc_block;
                            const float* bias_i = (const float*)bias_ + (i0 + i) * ldc + j0;
                            for(int64_t j = 0; j < nc; j++)
                                c_i[j] += bias_i[j]*beta;
                        }
                    }
                }
            }
        }, ntasks);
    }
}

class GemmOpImpl : public GemmOp
{
public:
    GemmOpImpl(bool transA_, bool transB_, double alpha_, double beta_)
    {
        transA = transA_;
        transB = transB_;
        alpha = alpha_;
        beta = beta_;
    }
    virtual std::string_view name() const CV_OVERRIDE { return "Gemm"; }
    virtual Op clone() const CV_OVERRIDE
    {
        return std::make_shared<GemmOpImpl>(transA, transB, alpha, beta);
    }

    virtual int minNumInputs() const CV_OVERRIDE { return 1; }
    virtual int maxNumInputs() const CV_OVERRIDE { return 3; }
    virtual int minNumOutputs() const CV_OVERRIDE { return 1; }
    virtual int maxNumOutputs() const CV_OVERRIDE { return 1; }

    virtual std::ostream& dumpAttrs(std::ostream& strm, int indent) const CV_OVERRIDE
    {
        prindent(strm, indent);
        strm << "transA: " << transA << ",\n";
        prindent(strm, indent);
        strm << "transB: " << transB << ",\n";
        prindent(strm, indent);
        strm << "alpha: " << alpha << ",\n";
        prindent(strm, indent);
        strm << "beta: " << beta << ",\n";
        return strm;
    }

    int inferType(int inptype0) const
    {
        CV_Assert(inptype0 == CV_32F || inptype0 == CV_16F);
        return CV_32F;
    }

    virtual bool supportType(int, int depth) const CV_OVERRIDE
    {
        return depth == CV_32F;
    }

    virtual bool alwaysSupportInplace() const CV_OVERRIDE
    {
        return false;
    }

    TensorSize inferShapes_(const TensorSize& inpsize, const TensorSize& wsize,
                            const TensorSize& biassize, bool transA_, bool transB_) const
    {
        CV_Assert(inpsize.ndims == 2);
        CV_Assert(wsize.ndims == 2);

        int64_t M = inpsize.size[transA_ ? 1 : 0];
        int64_t K1 = inpsize.size[transA_ ? 0 : 1];
        int64_t K2 = wsize.size[transB_ ? 1 : 0];
        int64_t N = wsize.size[transB_ ? 0 : 1];

        CV_Assert(K1 == K2);

        if (!biassize.empty()) {
            CV_Assert(biassize.ndims <= 2);
            int64_t bias_M = biassize.ndims == 2 ? biassize.size[1] : 1;
            int64_t bias_N = biassize.ndims > 0 ? biassize.size[biassize.ndims-1] : 1;
            CV_Assert(bias_M == 1 || bias_M == M);
            CV_Assert(bias_N == 1 || bias_N == N);
        }

        return TensorSize({M, N}, inpsize.layout);
    }

    virtual void setWeights(const Tensor& weights_, const Tensor& bias_, int accuracy) CV_OVERRIDE
    {
        CV_Assert(!weights_.empty());
        int wtype0 = weights_.type();
        CV_Assert(wtype0 == CV_32F || wtype0 == CV_16F);
        CV_Assert(accuracy == -1 || accuracy == CV_32F);
        int wtype = accuracy < 0 ? CV_32F : accuracy;

        wsize0 = weights_.size();
        weights_.copyTo(weights);
        biassize = bias_.size();

        if (!bias_.empty()) {
            bias_.convertTo(bias, CV_32F);
        } else {
            bias = Tensor();
        }
    }

    void fuseBatchNormWeights()
    {
        CV_Error(Error::StsNotImplemented, "");
    }

    virtual void fuseActivation(const Op& op) override
    {
        CV_Error(Error::StsNotImplemented, "");
        /*ElemwiseOp* activ_ptr = dynamic_cast<ElemwiseOp*>(op.get());
        CV_Assert(activ_ptr->maxNumInputs() == 1);
        CV_Assert(!activ);
        CV_Assert(activ_ptr != nullptr);
        activ = op;*/
    }

    virtual int64_t getFLOPS(const std::vector<SizeType> &inputs,
                             const std::vector<SizeType> &outputs) const CV_OVERRIDE
    {
        int ninputs = (int)inputs.size(), noutputs = (int)outputs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        CV_Assert(outputs.size() == 1);
        const TensorSize& inpsize = inputs[0].size;
        const TensorSize& outsize = outputs[0].size;
        CV_Assert(outsize.ndims == 2 && inpsize.ndims == 2);
        int64_t M = outsize.size[0], N = outsize.size[1];
        int64_t K = inpsize.size[transA ? 0 : 1];

        return M*N*K; // [TODO] take bias and possibly fused activation into account
    }

    virtual void inferTypes(const Net2& net, const Graph& graph,
                            const std::vector<Arg>& inpargs,
                            const std::vector<int>& inptypes,
                            const std::vector<Arg>& outargs,
                            std::vector<int>& outtypes) const CV_OVERRIDE
    {
        int ninputs = (int)inpargs.size(), noutputs = (int)outargs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        CV_Assert((int)inptypes.size() == ninputs);
        CV_Assert(noutputs == 1);

        outtypes.resize(1);
        outtypes[0] = inferType(inptypes[0]);
    }

    virtual void inferShapes(Net2& net, const Graph& graph,
                             const std::vector<Arg>& inpargs,
                             const std::vector<TensorSize>& inpshapes,
                             const std::vector<Arg>& outargs,
                             std::vector<TensorSize>& outshapes,
                             bool symbolic) const CV_OVERRIDE
    {
        int ninputs = (int)inpargs.size(), noutputs = (int)outargs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        CV_Assert(noutputs == 1);
        outshapes.resize(1);

        const TensorSize& inpsize = inpshapes[0];
        TensorSize wsize = ninputs > 1 ? inpshapes[1] : wsize0;
        TensorSize biassize = ninputs > 2 ? inpshapes[2] : bias.size();

        outshapes[0] = inferShapes_(inpsize, wsize, biassize, transA, transB);
    }

    virtual void forward(Net2& net, Graph& graph,
                        const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs,
                        std::vector<Buffer>& tempbufs) CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        const Tensor& inp = inputs[0];
        int inptype = inp.type(), outtype = inferType(inptype);
        TensorSize inpsize = inp.size();
        CV_Assert(inp.isContinuous());

        TensorSize outsize = inferShapes_(inpsize, ninputs > 1 ? inputs[1].size() : wsize0,
                                          ninputs > 2 ? inputs[2].size() : biassize, transA, transB);
        outputs.resize(1);
        Tensor& out = outputs[0];
        out.fitSameDevice(inp, outsize, outtype);
        CV_Assert(out.isContinuous());

        if (inp.empty())
            return;

        const Tensor& curr_w = ninputs > 1 ? inputs[1] : weights;
        const Tensor& curr_b = ninputs > 2 ? inputs[2] : bias;
        TensorSize curr_w_size = curr_w.size(), curr_b_size = curr_b.size();
        int64_t bias_step = curr_b_size.size[1];
        if (curr_b_size.ndims < 2) {
            CV_Assert(curr_b_size.ndims != 0);
            curr_b_size.ndims = 2;
            curr_b_size.size[1] = curr_b_size.size[0];
            curr_b_size.size[0] = 1;
            bias_step = 0;
        }

        /*if (batchNorm) {
            fuseBatchNormWeights();
            bias_data = fused_bias.ptr<float>();
        }*/

        gemm( transA, transB, alpha, beta,
              inpsize.size[0], inpsize.size[1], inp.type(), inp.data(), inpsize.size[0], 1,
              curr_w_size.size[0], curr_w_size.size[1], curr_w.type(), curr_w.data(), curr_w_size.size[0], 1,
              curr_b_size.size[0], curr_b_size.size[1], curr_b.type(), curr_b.data(), bias_step,
              out.type(), out.data(), outsize.size[0], 8 );
    }

    Tensor weights, bias;
    TensorSize wsize0, biassize;
};

GemmOp::~GemmOp() {}

Op GemmOp::create(bool transA, bool transB, double alpha, double beta)
{
    return std::make_shared<GemmOpImpl>(transA, transB, alpha, beta);
}

Arg gemm(Graph& graph, std::string_view opname,
         std::string_view outname, Arg A, Arg B, Arg bias,
         bool transA, bool transB, double alpha, double beta)
{
    Op op = GemmOp::create(transA, transB, alpha, beta);
    std::vector<Arg> inputs = {A, B};
    if (!bias.empty())
        inputs.push_back(bias);

    return graph->append(opname, op, outname, inputs);
}

}}
