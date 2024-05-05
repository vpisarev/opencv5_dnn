// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <math.h>

namespace cv { namespace dnn {

// C x 1 x Hk x Wk => C1 x Hk x Wk x C0
void repackDepthwiseConvWeights(const void* inpw__, int inptype_, void* outw__, int outtype_,
                                const TensorSize& wsize, int64_t C0_)
{
    CV_Assert(inptype_ == CV_32F || inptype_ == CV_16F);
    CV_Assert(outtype_ == CV_32F || outtype_ == CV_16F);

    int64_t C1_ = (wsize.size[0] + C0_ - 1)/C0_;
    parallel_for_(Range(0, (int)C1_), [&](const Range& r) {
        int inptype = inptype_, outtype = outtype_;
        size_t inp_esz = CV_ELEM_SIZE(inptype);
        size_t out_esz = CV_ELEM_SIZE(outtype);
        int64_t C = wsize.size[0];
        int64_t C0 = C0_, C1 = C1_;
        int64_t Hk = wsize.size[2], Wk = wsize.size[3];
        size_t inp_step_c = Hk*Wk;

        for (int64_t c1 = r.start; c1 < r.end; c1++) {
            const uint8_t* inpw_ = (const uint8_t*)inpw__ + c1*Hk*Wk*C0*inp_esz;
            uint8_t* outw_ = (uint8_t*)outw__ + c1*Hk*Wk*C0*out_esz;
            int64_t curr_C0 = std::min(C - c1*C0, C0);
            if (curr_C0 < C0)
                memset(outw_, 0, Hk*Wk*C0*out_esz);

            #define REPACK_WEIGHTS_CASE(inpT, outT) \
                (inptype == DataType<inpT>::depth && outtype == DataType<outT>::depth) { \
                    const inpT* inpw = (const inpT*)inpw_; \
                    outT* outw = (outT*)outw_; \
                    for (int64_t xy = 0; xy < Hk*Wk; xy++, inpw++, outw += C0) { \
                        for (int64_t c0 = 0; c0 < curr_C0; c0++) { \
                            outw[c0] = outT(inpw[inp_step_c*c0]); \
                        } \
                    } \
                }

            if REPACK_WEIGHTS_CASE(float, float)
            else if REPACK_WEIGHTS_CASE(float, hfloat)
            else if REPACK_WEIGHTS_CASE(hfloat, float)
            else if REPACK_WEIGHTS_CASE(hfloat, hfloat)
            else break;
        }
    });
}

static void conv2d_depthwise_32f(const void* inp__, void* out__, const ConvState& cs,
                                 const void* weights__, const float* scale__, const float* bias__)
{
    int nlanes_ = VTraits<v_float32>::vlanes();
    int C0_ = (int)cs.C0, C1_ = (int)cs.C1;

    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);

    int64_t NC = cs.N*C1_;
    int ksize_ = (int)(cs.Hk*cs.Wk);
    AutoBuffer<int64_t> local_ofstab_buf(ksize_);
    int64_t* local_ofstab = local_ofstab_buf.data();
    for (int k = 0; k < ksize_; k++)
        local_ofstab[k] = cs.ofstab[k]*C0_;

    parallel_for_(Range(0, (int)NC), [&](const Range& r)
    {
        int64_t nc0 = r.start, nc1 = r.end;
        int nlanes = nlanes_, C0 = C0_, C1 = C1_, C = (int)cs.C;
        int64_t Hi = cs.Hi, Wi = cs.Wi;
        int64_t H = cs.H, W = cs.W;
        int64_t iplanesize = Hi*Wi*C0;
        int64_t planesize = H*W*C0;
        int64_t SY = cs.SY, SX = cs.SX;
        int64_t DY = cs.DY, DX = cs.DX;
        int64_t pad_y0 = cs.pad_y0, pad_x0 = cs.pad_x0;
        int64_t pad_y1 = cs.pad_y1, pad_x1 = cs.pad_x1;
        int64_t inner_y0 = cs.inner_y0, inner_y1 = cs.inner_y1;
        int64_t inner_x0 = cs.inner_x0, inner_x1 = cs.inner_x1;
        int ksize = ksize_;
        const int* yxtab = cs.yxtab;
        const int64_t* ofstab = local_ofstab;
        const float* scale_ = scale__;
        const float* bias_ = bias__;
        AutoBuffer<float> buf(C0*3);
        float* sum = buf.data();
        float* scale = sum + C0;
        float* bias = scale + C0;

        FastActivation fastActivation = cs.fastActivation;
        const float* activParams = cs.activParams;
        ElemwiseOp::activ_t activation = cs.activation;
        float maxval = fastActivation == ACTIV_CLIP ? activParams[1] : FLT_MAX;
        float alpha = fastActivation == ACTIV_LEAKY_RELU ? activParams[0] :
                    fastActivation == ACTIV_NONE ? 1.f : 0.f;
        v_float32 v_maxval = vx_setall_f32(maxval);
        v_float32 v_alpha = vx_setall_f32(alpha);

        const float* inp = (const float*)inp__ + nc0*iplanesize;
        float* out = (float*)out__ + nc0*planesize;
        v_float32 z = vx_setzero_f32();

        for (int64_t nc = nc0; nc < nc1; nc++, inp += iplanesize) {
            int64_t n = nc / C1;
            int c0 = (int)((nc - n*C1)*C0), C0_curr = std::min(C0, C - c0);
            const float* weights = (const float*)weights__ + (c0/C0)*ksize*C0;

            for (int64_t c = 0; c < C0; c++) {
                scale[c] = scale_ ? scale_[c0 + c] : 1.f;
                bias[c] = bias_ ? bias_[c0 + c] : 0.f;
            }

            for (int64_t y0 = 0; y0 < H; y0++, out += W*C0) {
                //int64_t x0 = 0, x1 = W;
                int64_t x0 = 0, x1 = y0 >= inner_y0 && y0 < inner_y1 ? inner_x0 : W;
                int64_t yi_ = y0*SY - pad_y0;
                for(;;) {
                    if (nlanes == C0) {
                        v_float32 sc0 = vx_load(scale), b0 = vx_load(bias);
                        for (; x0 < x1; x0++) {
                            int64_t xi_ = x0*SX - pad_x0;
                            v_float32 s0 = z;
                            for (int k = 0; k < ksize; k++) {
                                int64_t yi = yi_ + yxtab[k*2];
                                int64_t xi = xi_ + yxtab[k*2+1];
                                v_float32 v0, w0;
                                if ((uint64_t)yi >= (uint64_t)Hi || (uint64_t)xi >= (uint64_t)Wi)
                                    continue;
                                v0 = vx_load(inp + (yi*Wi + xi)*C0);
                                w0 = vx_load(weights + k*C0);
                                s0 = v_fma(v0, w0, s0);
                            }
                            s0 = v_fma(s0, sc0, b0);
                            s0 = v_min(v_select(v_ge(s0, z), s0, v_mul(s0, v_alpha)), v_maxval);
                            vx_store(out + x0*C0, s0);
                        }
                    } else {
                        for (; x0 < x1; x0++) {
                            int64_t xi_ = x0*SX - pad_x0;
                            for (int c = 0; c < C0; c += nlanes*2) {
                                v_float32 s0 = z, s1 = z;
                                for (int k = 0; k < ksize; k++) {
                                    int64_t yi = yi_ + yxtab[k*2];
                                    int64_t xi = xi_ + yxtab[k*2+1];
                                    v_float32 v0, v1, w0, w1;
                                    if ((uint64_t)yi >= (uint64_t)Hi || (uint64_t)xi >= (uint64_t)Wi)
                                        continue;
                                    int64_t ofs_k = (yi*Wi + xi)*C0 + c;
                                    int64_t ofs_w = k*C0;
                                    v0 = vx_load(inp + ofs_k);
                                    v1 = vx_load(inp + ofs_k + nlanes);
                                    w0 = vx_load(weights + ofs_w);
                                    w1 = vx_load(weights + ofs_w + nlanes);
                                    s0 = v_fma(v0, w0, s0);
                                    s1 = v_fma(v1, w1, s1);
                                }
                                s0 = v_fma(s0, vx_load(scale + c), vx_load(bias + c));
                                s1 = v_fma(s1, vx_load(scale + c + nlanes), vx_load(bias + c + nlanes));
                                s0 = v_min(v_select(v_ge(s0, z), s0, v_mul(s0, v_alpha)), v_maxval);
                                s1 = v_min(v_select(v_ge(s1, z), s1, v_mul(s1, v_alpha)), v_maxval);
                                vx_store(out + x0*C0 + c, s0);
                                vx_store(out + x0*C0 + c + nlanes, s1);
                            }
                        }
                    }
                    if (x0 == W)
                        break;
                    x1 = inner_x1;
                    if (nlanes == C0) {
                        v_float32 sc0 = vx_load(scale), b0 = vx_load(bias);
                        for (; x0 < x1; x0++) {
                            int64_t xi_ = x0*SX - pad_x0;
                            const float* inp_xi = inp + (Wi*yi_ + xi_)*C0;

                            v_float32 s0 = z;
                            for (int k = 0; k < ksize; k++) {
                                v_float32 v0 = vx_load(inp_xi + ofstab[k]);
                                v_float32 w0 = vx_load(weights + k*C0);
                                s0 = v_fma(v0, w0, s0);
                            }
                            s0 = v_fma(s0, sc0, b0);
                            s0 = v_min(v_select(v_ge(s0, z), s0, v_mul(s0, v_alpha)), v_maxval);
                            vx_store(out + x0*C0, s0);
                        }
                    } else if (nlanes*2 == C0) {
                        v_float32 sc0 = vx_load(scale), sc1 = vx_load(scale + nlanes);
                        v_float32 b0 = vx_load(bias), b1 = vx_load(bias + nlanes);
                        for (; x0 < x1; x0++) {
                            int64_t xi_ = x0*SX - pad_x0;
                            const float* inp_xi = inp + (Wi*yi_ + xi_)*C0;

                            v_float32 s0 = z, s1 = z;
                            for (int k = 0; k < ksize; k++) {
                                int64_t ofs_k = ofstab[k], ofs_w = k*C0;
                                v_float32 v0 = vx_load(inp_xi + ofs_k);
                                v_float32 v1 = vx_load(inp_xi + ofs_k + nlanes);
                                v_float32 w0 = vx_load(weights + ofs_w);
                                v_float32 w1 = vx_load(weights + ofs_w + nlanes);
                                s0 = v_fma(v0, w0, s0);
                                s1 = v_fma(v1, w1, s1);
                            }
                            s0 = v_fma(s0, sc0, b0);
                            s1 = v_fma(s1, sc1, b1);
                            s0 = v_min(v_select(v_ge(s0, z), s0, v_mul(s0, v_alpha)), v_maxval);
                            s1 = v_min(v_select(v_ge(s1, z), s1, v_mul(s1, v_alpha)), v_maxval);
                            vx_store(out + x0*C0, s0);
                            vx_store(out + x0*C0 + nlanes, s1);
                        }
                    } else {
                        for (; x0 < x1; x0++) {
                            int64_t xi_ = x0*SX - pad_x0;
                            for (int c = 0; c < C0; c += nlanes*4) {
                                const float* inp_xi = inp + (Wi*yi_ + xi_)*C0 + c;
                                v_float32 s0 = z, s1 = z, s2 = z, s3 = z;
                                for (int k = 0; k < ksize; k++) {
                                    int64_t ofs_k = ofstab[k], ofs_w = k*C0 + c;
                                    v_float32 v0 = vx_load(inp_xi + ofs_k);
                                    v_float32 v1 = vx_load(inp_xi + ofs_k + nlanes);
                                    v_float32 v2 = vx_load(inp_xi + ofs_k + nlanes*2);
                                    v_float32 v3 = vx_load(inp_xi + ofs_k + nlanes*3);
                                    v_float32 w0 = vx_load(weights + ofs_w);
                                    v_float32 w1 = vx_load(weights + ofs_w + nlanes);
                                    v_float32 w2 = vx_load(weights + ofs_w + nlanes*2);
                                    v_float32 w3 = vx_load(weights + ofs_w + nlanes*3);
                                    s0 = v_fma(v0, w0, s0);
                                    s1 = v_fma(v1, w1, s1);
                                    s2 = v_fma(v2, w2, s2);
                                    s3 = v_fma(v3, w3, s3);
                                }
                                s0 = v_fma(s0, vx_load(scale + c), vx_load(bias + c));
                                s1 = v_fma(s1, vx_load(scale + c + nlanes), vx_load(bias + c + nlanes));
                                s2 = v_fma(s2, vx_load(scale + c + nlanes*2), vx_load(bias + c + nlanes*2));
                                s3 = v_fma(s3, vx_load(scale + c + nlanes*3), vx_load(bias + c + nlanes*3));
                                s0 = v_min(v_select(v_ge(s0, z), s0, v_mul(s0, v_alpha)), v_maxval);
                                s1 = v_min(v_select(v_ge(s1, z), s1, v_mul(s1, v_alpha)), v_maxval);
                                s2 = v_min(v_select(v_ge(s2, z), s2, v_mul(s2, v_alpha)), v_maxval);
                                s3 = v_min(v_select(v_ge(s3, z), s3, v_mul(s3, v_alpha)), v_maxval);
                                vx_store(out + x0*C0 + c, s0);
                                vx_store(out + x0*C0 + c + nlanes, s1);
                                vx_store(out + x0*C0 + c + nlanes*2, s2);
                                vx_store(out + x0*C0 + c + nlanes*3, s3);
                            }
                        }
                    }
                    x1 = W;
                }
            }

            if (activation) {
                activation(out - planesize, out - planesize, planesize, activParams);
            }
        }
    });
}

depthwise_conv2d_t getDepthwiseConv2DFunc(int depth)
{
    depthwise_conv2d_t func = depth == CV_32F ? conv2d_depthwise_32f : nullptr;
    return func;
}

}}
