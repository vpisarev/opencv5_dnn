// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/net2_impl.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv { namespace dnn {

static void initConv2DTables(const ConvState& cs,
                             std::vector<int32_t>& ofsbuf_,
                             std::vector<int32_t>& ofs0_,
                             std::vector<int32_t*>& ofsptrs_,
                             std::vector<uint8_t>& mask_)
{
    int64_t Hk = cs.Hk, Wk = cs.Wk;
    int64_t DY = cs.DY, DX = cs.DX, SY = cs.SY, SX = cs.SX;
    int64_t pad_y0 = cs.pad_y0, pad_x0 = cs.pad_x0;
    int64_t Hi = cs.Hi, Wi = cs.Wi, H = cs.H, W = cs.W;
    int64_t C0 = cs.C0, K0 = C0, C1 = cs.C1, K1 = cs.K1;
    int64_t ngroups = cs.ngroups, C1g = C1/ngroups, K1g = K1/ngroups;
    int64_t inner_y0 = cs.inner_y0, inner_y1 = cs.inner_y1;
    int64_t inner_x0 = cs.inner_x0, inner_x1 = cs.inner_x1;

    mask_.resize(H*W);
    ofs0_.resize(H*W);
    ofsptrs_.resize(H*W);

    int64_t ofs_blocksize = C1g*Hk*Wk;
    bool have_inner = inner_y0 < inner_y1 && inner_x0 < inner_x1;

    int64_t nblocks = have_inner ? 1 + (inner_y0 + (H - inner_y1))*W +
        (inner_y1 - inner_y0)*(inner_x0 + W - inner_x1) : W*H;

    ofsbuf_.resize(ofs_blocksize*nblocks);
    int32_t* ofsbuf = ofsbuf_.data();

    if (have_inner) {
        for (int64_t c = 0, k = 0; c < C1g; c++) {
            for (int64_t dy = 0; dy < Hk; dy++) {
                int64_t yi = dy*DY;
                for (int64_t dx = 0; dx < Wk; dx++, k++) {
                    int64_t xi = dx*DX;
                    ofsbuf[k] = (int32_t)(((c*Hi + yi)*Wi + xi)*C0);
                }
            }
        }
    }

    parallel_for_(Range(0, (int)H), [&](const Range& r) {
        uint8_t* mask = mask_.data();
        int32_t* ofs0 = ofs0_.data();
        int32_t** ofsptrs = ofsptrs_.data();
        int64_t curr_block = 1;
        if (have_inner) {
            curr_block += std::min((int64_t)r.start, inner_y0)*W;
            curr_block += std::min(std::max(r.start - inner_y0, (int64_t)0),
                                   inner_y1 - inner_y0)*(inner_x0 + W - inner_x1);
            curr_block += std::max(r.start - inner_y1, (int64_t)0)*W;
        } else {
            curr_block = r.start*W;
        }
        for (int64_t y0 = r.start; y0 < r.end; y0++) {
            int64_t yi_ = y0*SY - pad_y0;
            bool y_inside = inner_y0 <= y0 && y0 < inner_y1;

            for (int64_t x0 = 0; x0 < W; x0++) {
                int64_t xi_ = x0*SX - pad_x0;
                bool x_inside = inner_x0 <= x0 && x0 < inner_x1;
                uint8_t m = (uint8_t)(y_inside & x_inside);

                mask[y0*W + x0] = m;

                if (m) {
                    ofs0[y0*W + x0] = (int32_t)((yi_*Wi + xi_)*C0);
                    ofsptrs[y0*W + x0] = ofsbuf;
                } else {
                    ofs0[y0*W + x0] = 0;
                    int32_t* ofsptr = ofsbuf + curr_block*ofs_blocksize;
                    ofsptrs[y0*W + x0] = ofsptr;
                    curr_block++;

                    for (int64_t c = 0, k = 0; c < C1g; c++) {
                        for (int64_t dy = 0; dy < Hk; dy++) {
                            int64_t yi = yi_ + dy*DY;
                            bool yi_inside = 0 <= yi && yi < Hi;

                            for (int64_t dx = 0; dx < Wk; dx++, k++) {
                                int64_t xi = xi_ + dx*DX;
                                bool xi_inside = 0 <= xi && xi < Wi;
                                ofsptr[k] = (yi_inside & xi_inside) ?
                                    (int32_t)(((c*Hi + yi)*Wi + xi)*C0) : INT_MIN/2;
                            }
                        }
                    }
                }
            }
        }
    });
}

template<typename _InpT, typename _OutT> void
repackConvWeights_(const _InpT* inpw_, _OutT* outw_,
                   int64_t inp_step_c, int64_t inp_step_k, int64_t HkWk,
                   int64_t C0, int64_t K0, int64_t curr_C0, int64_t curr_K0)
{
    const _InpT* inpw = inpw_;
    _OutT* outw = outw_;
    for (int64_t xy = 0; xy < HkWk; xy++, inpw++, outw += C0*K0) {
        for (int64_t c0 = 0; c0 < curr_C0; c0++) {
            for (int64_t k0 = 0; k0 < curr_K0; k0++) {
                outw[c0*K0 + k0] = _OutT(inpw[inp_step_k*k0 + inp_step_c*c0]);
            }
        }
    }
}


// K x (C/ngroups) x Hk x Wk => K1 x C1/ngroups x Hk x Wk x C0 x K0,
// where K0 == C0
static void repackConvWeights(const void* inpw__, int inptype_, void* outw__, int outtype_,
                              const TensorSize& wsize, int64_t C0_)
{
    CV_Assert(inptype_ == CV_32F || inptype_ == CV_16F);
    CV_Assert(outtype_ == CV_32F || outtype_ == CV_16F);

    int64_t K1_ = (wsize.size[0] + C0_ - 1)/C0_;
    parallel_for_(Range(0, (int)K1_), [&](const Range& r) {
        int inptype = inptype_, outtype = outtype_;
        size_t inp_esz = CV_ELEM_SIZE(inptype);
        size_t out_esz = CV_ELEM_SIZE(outtype);
        int64_t C0 = C0_, K0 = C0_;
        int64_t K = wsize.size[0], Cg = wsize.size[1];
        int64_t K1 = K1_, C1g = (Cg + C0 - 1)/C0;
        int64_t Hk = wsize.size[2], Wk = wsize.size[3];
        size_t inp_step_c = Hk*Wk, inp_step_k = Cg*Hk*Wk;
        size_t out_microplane_size = Hk*Wk*C0*K0*out_esz;

        for (int64_t k1 = r.start; k1 < r.end; k1++) {
            int64_t curr_K0 = std::min(K - k1*K0, K0);
            for (int64_t c1g = 0; c1g < C1g; c1g++) {
                uint8_t* inpw_ = (uint8_t*)inpw__ + (k1*K0*inp_step_k + c1g*C0*inp_step_c)*inp_esz;
                uint8_t* outw_ = (uint8_t*)outw__ + (k1*C1g + c1g)*out_microplane_size;
                int64_t curr_C0 = std::min(Cg - c1g*C0, C0);
                if (curr_K0 != K0 || curr_C0 != C0)
                    memset(outw_, 0, out_microplane_size);

                if (inptype == CV_32F && outtype == CV_32F)
                    repackConvWeights_((const float*)inpw_, (float*)outw_, inp_step_c,
                                       inp_step_k, Hk*Wk, C0, K0, curr_C0, curr_K0);
                else if (inptype == CV_32F && outtype == CV_16F)
                    repackConvWeights_((const float*)inpw_, (hfloat*)outw_, inp_step_c,
                                       inp_step_k, Hk*Wk, C0, K0, curr_C0, curr_K0);
                else if (inptype == CV_16F && outtype == CV_32F)
                    repackConvWeights_((const hfloat*)inpw_, (float*)outw_, inp_step_c,
                                       inp_step_k, Hk*Wk, C0, K0, curr_C0, curr_K0);
                else if (inptype == CV_16F && outtype == CV_16F)
                    repackConvWeights_((const hfloat*)inpw_, (hfloat*)outw_, inp_step_c,
                                       inp_step_k, Hk*Wk, C0, K0, curr_C0, curr_K0);
                else break;
            }
        }
    });
}

static void conv2d_32f(const void* inp__, const void* residual__, void* out__,
                       const ConvState& cs, const void* weights__,
                       const float* scale__, const float* bias__,
                       const int32_t* ofs0__, const int32_t** ofsptrs__,
                       const uint8_t* mask__)
{
    int nlanes_ = VTraits<v_float32>::vlanes();
    int C0_ = (int)cs.C0;

    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);
    CV_Assert(cs.activation == nullptr || cs.fastActivation == ACTIV_NONE);

    int64_t NK1 = cs.N*cs.K1;

    parallel_for_(Range(0, (int)NK1), [&](const Range& r) {
        const float* scale_ = scale__;
        const float* bias_ = bias__;
        const int32_t* ofs0_ = ofs0__;
        const int32_t** ofsptrs_ = ofsptrs__;
        const uint8_t* mask_ = mask__;
        constexpr int64_t BLOCK_SIZE = 10;
        int64_t nk0 = r.start, nk1 = r.end;
        int nlanes = nlanes_, C0 = C0_, K0 = C0;
        int64_t Hi = cs.Hi, Wi = cs.Wi;
        int64_t H0 = cs.H, W0 = cs.W;
        int64_t iplanesize = Hi*Wi;
        int64_t planesize = H0*W0;
        int64_t SY = cs.SY, SX = cs.SX;
        int64_t DY = cs.DY, DX = cs.DX;
        int64_t Hk = cs.Hk, Wk = cs.Wk;
        int64_t pad_y0 = cs.pad_y0, pad_x0 = cs.pad_x0;
        int64_t C1 = cs.C1, K1 = cs.K1;
        int64_t ngroups = cs.ngroups, K1g = K1/ngroups, C1g = C1/ngroups;
        int64_t nC = C1g*Hk*Wk*C0*K0;
        AutoBuffer<float> sumbuf(BLOCK_SIZE*K0*3);
        float* sum = sumbuf.data();
        float* scale = sum + BLOCK_SIZE*K0;
        float* bias = sum + BLOCK_SIZE*K0*2;
        const float* inptrs[BLOCK_SIZE];
        const int32_t* ofsptrs[BLOCK_SIZE];
        FastActivation fastActivation = cs.fastActivation;
        const float* activParams = cs.activParams;
        ElemwiseOp::activ_t activation = cs.activation;
        float maxval = fastActivation == ACTIV_CLIP ? activParams[1] : FLT_MAX;
        float alpha = fastActivation == ACTIV_LEAKY_RELU ? activParams[0] :
                    fastActivation == ACTIV_NONE ? 1.f : 0.f;

        for (int64_t j = 0; j < BLOCK_SIZE*K0; j++) {
            scale[j] = 1.f;
            bias[j] = 0.f;
        }

        for (int64_t nk = nk0; nk < nk1; nk++) {
            int64_t n = nk/K1, k1 = nk - n*K1;
            int64_t g = k1/K1g;
            float* out = (float*)out__ + nk*planesize*K0;
            const float* inp0 = (const float*)inp__ + (n*C1 + g*C1g)*iplanesize*C0;
            const float* resptr = residual__ ? (const float*)residual__ + nk*planesize*K0 : nullptr;
            const float* wptr = (const float*)weights__ + k1*nC;

            if (scale_) {
                for (int64_t b = 0; b < BLOCK_SIZE; b++)
                    for (int64_t k = 0; k < K0; k++)
                        scale[b*K0 + k] = scale_[k1*K0 + k];
            }

            if (bias_) {
                for (int64_t b = 0; b < BLOCK_SIZE; b++)
                    for (int64_t k = 0; k < K0; k++)
                        bias[b*K0 + k] = bias_[k1*K0 + k];
            }

            for (int64_t xy0 = 0; xy0 < W0*H0; xy0 += BLOCK_SIZE, out += K0*BLOCK_SIZE,
                                               resptr += (resptr ? K0*BLOCK_SIZE : 0)) {
                int64_t j = 0, blocksize = std::min(W0*H0 - xy0, BLOCK_SIZE);

                for (; j < blocksize; j++) {
                    inptrs[j] = inp0 + ofs0_[xy0 + j];
                    ofsptrs[j] = ofsptrs_[xy0 + j];
                }

                if (j < BLOCK_SIZE) {
                    const float* last_inptr = inptrs[blocksize-1];
                    const int32_t* last_ofsptr = ofsptrs[blocksize-1];
                    for (; j < BLOCK_SIZE; j++) {
                        inptrs[j] = last_inptr;
                        ofsptrs[j] = last_ofsptr;
                    }
                }

                for (int64_t i = 0; i < BLOCK_SIZE*K0; i++)
                    sum[i] = 0.f;

                for (int64_t c1 = 0, i = 0; c1 < nC; c1 += K0*C0, i++) {
                    for (j = 0; j < BLOCK_SIZE; j++) {
                        int32_t ofs_ij = ofsptrs[j][i];
                        const float* x = &inptrs[j][std::max(ofs_ij, 0)];
                        float mij = (float)(ofs_ij >= 0);
                        for (int64_t c0 = 0; c0 < C0; c0++) {
                            float xc = x[c0]*mij;
                            for (int64_t k = 0; k < K0; k++) {
                                float w = wptr[c1 + c0*K0 + k];
                                sum[K0*j + k] += xc*wptr[c1 + c0*K0 + k];
                            }
                        }
                    }
                }

                if (activation) {
                    if (resptr) {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            sum[j] = v;
                        }
                    } else {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            sum[j] = v;
                        }
                    }
                    activation(sum, out, blocksize*K0, activParams);
                } else {
                    if (resptr) {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    } else {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    }
                }
            }
        }
    });
}

static void conv2d_1x1_32f(const void* inp__, const void* residual__, void* out__,
                           const ConvState& cs, const void* weights__,
                           const float* scale__, const float* bias__,
                           const int32_t*, const int32_t**, const uint8_t*)
{
    int nlanes_ = VTraits<v_float32>::vlanes();
    int C0_ = (int)cs.C0;

    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);
    CV_Assert(cs.activation == nullptr || cs.fastActivation == ACTIV_NONE);
    CV_Assert(cs.Hk == 1 && cs.Wk == 1);
    CV_Assert(cs.pad_y0 == 0 && cs.pad_x0 == 0 && cs.pad_y1 == 0 && cs.pad_x1 == 0);

    int64_t NK1 = cs.N*cs.K1;

    parallel_for_(Range(0, (int)NK1), [&](const Range& r) {
        const float* scale_ = scale__;
        const float* bias_ = bias__;
        constexpr int64_t BLOCK_SIZE = 10;
        int64_t nk0 = r.start, nk1 = r.end;
        int nlanes = nlanes_, C0 = C0_, K0 = C0;
        int64_t Hi = cs.Hi, Wi = cs.Wi;
        int64_t H0 = cs.H, W0 = cs.W;
        int64_t iplanesize = Hi*Wi;
        int64_t planesize = H0*W0;
        int64_t SY = cs.SY, SX = cs.SX;
        int64_t Hk = cs.Hk, Wk = cs.Wk;
        int64_t C1 = cs.C1, K1 = cs.K1;
        int64_t ngroups = cs.ngroups, K1g = K1/ngroups, C1g = C1/ngroups;
        int64_t nC = C1g*C0*K0;
        AutoBuffer<float> sumbuf(BLOCK_SIZE*K0*3);
        float* sum = sumbuf.data();
        float* scale = sum + BLOCK_SIZE*K0;
        float* bias = sum + BLOCK_SIZE*K0*2;
        const float* inptrs[BLOCK_SIZE];
        FastActivation fastActivation = cs.fastActivation;
        const float* activParams = cs.activParams;
        ElemwiseOp::activ_t activation = cs.activation;
        float maxval = fastActivation == ACTIV_CLIP ? activParams[1] : FLT_MAX;
        float alpha = fastActivation == ACTIV_LEAKY_RELU ? activParams[0] :
                    fastActivation == ACTIV_NONE ? 1.f : 0.f;
        bool S1 = SY == 1 && SX == 1;

        for (int64_t j = 0; j < BLOCK_SIZE*K0; j++) {
            scale[j] = 1.f;
            bias[j] = 0.f;
        }

        for (int64_t nk = nk0; nk < nk1; nk++) {
            int64_t n = nk/K1, k1 = nk - n*K1;
            int64_t g = k1/K1g;
            float* out = (float*)out__ + nk*planesize*K0;
            const float* inp0 = (const float*)inp__ + (n*C1 + g*C1g)*iplanesize*C0;
            const float* resptr = residual__ ? (const float*)residual__ + nk*planesize*K0 : nullptr;
            const float* wptr = (const float*)weights__ + k1*nC;

            if (scale_) {
                for (int64_t b = 0; b < BLOCK_SIZE; b++)
                    for (int64_t k = 0; k < K0; k++)
                        scale[b*K0 + k] = scale_[k1*K0 + k];
            }

            if (bias_) {
                for (int64_t b = 0; b < BLOCK_SIZE; b++)
                    for (int64_t k = 0; k < K0; k++)
                        bias[b*K0 + k] = bias_[k1*K0 + k];
            }

            int64_t yiWi = 0, xi = 0;
            for (int64_t xy0 = 0; xy0 < W0*H0; xy0 += BLOCK_SIZE, out += K0*BLOCK_SIZE,
                                               resptr += (resptr ? K0*BLOCK_SIZE : 0))
            {
                int64_t j = 0, blocksize = std::min(W0*H0 - xy0, BLOCK_SIZE);

                if (S1) {
                    for (; j < blocksize; j++) {
                        inptrs[j] = inp0 + (xy0 + j)*C0;
                    }
                } else {
                    for (; j < blocksize; j++) {
                        inptrs[j] = inp0 + (yiWi + xi)*C0;
                        if ((xi += SX) >= Wi) {
                            yiWi += Wi*SY;
                            xi = 0;
                        }
                    }
                }

                if (j < BLOCK_SIZE) {
                    const float* last_inptr = inptrs[blocksize-1];
                    for (; j < BLOCK_SIZE; j++)
                        inptrs[j] = last_inptr;
                }

                for (int64_t i = 0; i < BLOCK_SIZE*K0; i++)
                    sum[i] = 0.f;

                for (int64_t c1 = 0, i = 0; c1 < nC; c1 += K0*C0, i++) {
                    int64_t ofs_ij = i*iplanesize*C0;
                    for (j = 0; j < BLOCK_SIZE; j++) {
                        const float* x = &inptrs[j][ofs_ij];
                        for (int64_t c0 = 0; c0 < C0; c0++) {
                            float xc = x[c0];
                            for (int64_t k = 0; k < K0; k++) {
                                float w = wptr[c1 + c0*K0 + k];
                                sum[K0*j + k] += xc*wptr[c1 + c0*K0 + k];
                            }
                        }
                    }
                }

                if (activation) {
                    if (resptr) {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            sum[j] = v;
                        }
                    } else {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            sum[j] = v;
                        }
                    }
                    activation(sum, out, blocksize*K0, activParams);
                } else {
                    if (resptr) {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    } else {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    }
                }
            }
        }
    });
}


typedef void (*conv_func_t)(const void* inp, const void* residual, void* out,
                            const ConvState& cs, const void* weights,
                            const float* scale, const float* bias,
                            const int32_t* ofs0, const int32_t** ofsptrs,
                            const uint8_t* mask);

class ConvOpImpl : public ConvOp
{
public:
    ConvOpImpl(const ConvParams& convparams_)
    {
        params = convparams_;
        add_residual = false;
    }
    virtual std::string_view name() const CV_OVERRIDE { return "Conv"; }
    virtual Op clone() const CV_OVERRIDE
    {
        return std::make_shared<ConvOpImpl>(params);
    }

    virtual int minNumInputs() const CV_OVERRIDE { return 1 + (add_residual ? 1 : 0); }
    virtual int maxNumInputs() const CV_OVERRIDE { return 3 + (add_residual ? 1 : 0); }
    virtual int minNumOutputs() const CV_OVERRIDE { return 1; }
    virtual int maxNumOutputs() const CV_OVERRIDE { return 1; }

    virtual std::ostream& dumpAttrs(std::ostream& strm, int indent) const CV_OVERRIDE
    {
        prindent(strm, indent);
        strm << "ngroup: " << params.ngroups << ",\n";

        /*prindent(strm, indent);
        strm << "ksizes: [";
        for (int k = 0; k < wsize0.ndims; k++)
            strm << (k > 0 ? ", " : "") << wsize0.size[k];
        strm << "],\n";*/

        prindent(strm, indent);
        strm << "dilations: [";
        for (size_t k = 0; k < params.dilations.size(); k++)
            strm << (k > 0 ? ", " : "") << params.dilations[k];
        strm << "],\n";

        prindent(strm, indent);
        strm << "pads: [";
        for (size_t k = 0; k < params.pads.size(); k++)
            strm << (k > 0 ? ", " : "") << params.pads[k];
        strm << "],\n";

        prindent(strm, indent);
        strm << "strides: [";
        for (size_t k = 0; k < params.strides.size(); k++)
            strm << (k > 0 ? ", " : "") << params.strides[k];
        strm << "],\n";

        if (batchNorm) {
            prindent(strm, indent);
            strm << "batch_norm: true,\n";
        }

        if (add_residual) {
            prindent(strm, indent);
            strm << "add_residual: true,\n";
        }

        if (activ) {
            prindent(strm, indent);
            strm << "activation: " << activ->name() << ",\n";
        }

        return strm;
    }

    int inferType(int inptype0) const
    {
        return inptype0;
    }

    virtual bool supportType(int, int depth) const CV_OVERRIDE
    {
        return depth == CV_32F;
    }

    virtual bool alwaysSupportInplace() const CV_OVERRIDE
    {
        return false;
    }

    virtual int supportBlockLayout(int input, int ninputs) const CV_OVERRIDE
    {
        return input == 0 || (add_residual && input == ninputs-1) ? 1 : -1;
    }

    virtual void setWeights(const Tensor& weights_, const Tensor& bias_,
                            int64_t C0, int accuracy) CV_OVERRIDE
    {
        CV_Assert(!weights_.empty());
        int wtype0 = weights_.type();
        CV_Assert(wtype0 == CV_32F || wtype0 == CV_16F);
        CV_Assert(accuracy == -1 || accuracy == CV_32F);
        int wtype = accuracy < 0 ? CV_32F : accuracy;

        wsize0 = weights_.size();
        TensorSize wsize1 = wsize0;
        bool depthwise = params.ngroups == wsize0.size[0] && wsize0.size[1] == 1;

        if (depthwise) {
            wsize1.layout = LAYOUT_NCHWc;
            wsize1.C = wsize1.size[0];
            wsize1.size[0] = (wsize1.size[0] + C0 - 1)/C0;
            for (int i = 2; i < wsize1.ndims; i++)
                wsize1.size[i-1] = wsize1.size[i];
            wsize1.size[wsize1.ndims-1] = C0;
            weights.fit(wsize1, wtype);

            repackDepthwiseConvWeights(weights_.data(), wtype0, weights.data(), wtype, wsize0, C0);
        } else {
            wsize1.ndims += 2;
            wsize1.size[wsize1.ndims-1] = wsize1.size[wsize1.ndims-2] = C0;
            wsize1.size[0] = (wsize1.size[0] + C0 - 1)/C0;
            wsize1.size[1] = (wsize1.size[1] + C0 - 1)/C0;
            weights.fit(wsize1, wtype);

            repackConvWeights(weights_.data(), wtype0, weights.data(), wtype, wsize0, C0);
        }

        if (!bias_.empty()) {
            CV_Assert(bias_.isContinuous() && bias_.total() == wsize0.size[0]);
            bias_.convertTo(bias, CV_32F);
        }
    }

    void fuseBatchNormWeights()
    {
        BatchNormOp* bn = dynamic_cast<BatchNormOp*>(batchNorm.get());
        CV_Assert(bn != nullptr);
        const Tensor &bn_scale = bn->scale, &bn_bias = bn->bias;

        CV_Assert(bn_scale.isContinuous() && bn_bias.isContinuous());
        CV_Assert(bn_scale.type() == CV_32F && bn_bias.type() == CV_32F);
        CV_Assert(bn_scale.total() == bn_bias.total());
        int64_t K = bn_scale.total();
        CV_Assert(bias.empty() || (bias.type() == CV_32F && bias.total() == K));
        const float* bias_data = bias.ptr<float>();

        fused_scale.fit(TensorSize(1, &K), CV_32F);
        fused_bias.fit(TensorSize(1, &K), CV_32F);

        const float* bn_scale_data = bn_scale.ptr<float>();
        const float* bn_bias_data = bn_bias.ptr<float>();
        float* fused_scale_data = fused_scale.ptr<float>();
        float* fused_bias_data = fused_bias.ptr<float>();

        // (sum(x*w) + bias)*bn_scale + bn_bias => sum(x*w)*fused_scale + fused_bias,
        // where fused_scale = bn_scale and fused_bias = bias*bn_scale + bn_bias.
        for (int64_t i = 0; i < K; i++) {
            fused_scale_data[i] = bn_scale_data[i];
            fused_bias_data[i] = (bias_data ? bn_scale_data[i]*bias_data[i] : 0.f) + bn_bias_data[i];
        }
    }

    virtual bool fuseBatchNorm(const Op& op) override
    {
        BatchNormOp* bn = dynamic_cast<BatchNormOp*>(op.get());
        if (batchNorm || !bn)
            return false;
        batchNorm = op;
        return true;
    }

    virtual bool fuseActivation(const Op& op) override
    {
        ElemwiseOp* activ_ptr = dynamic_cast<ElemwiseOp*>(op.get());
        if (activ || activ_ptr->maxNumInputs() != 1 || !activ_ptr || !activ_ptr->getActivation(CV_32F))
            return false;
        activ = op;
        return true;
    }

    virtual int64_t getFLOPS(const std::vector<SizeType> &inputs,
                             const std::vector<SizeType> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 2);
        CV_Assert(outputs.size() == 1);
        // probably, there should be a coefficient in the case of complex reduction functions
        TensorSize inpsize = inputs[0].size, wsize = inputs[1].size;
        int64_t C = inpsize.size[1]*inpsize.size[inpsize.ndims-1];
        int64_t ksize = (int64_t)wsize.total();
        return (int64_t)((inputs[0].size.total()/C)*ksize/params.ngroups);
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
        if (add_residual)
            ninputs--;

        const TensorSize& inpsize = inpshapes[0];
        TensorSize wsize = ninputs > 1 ? inpshapes[1] : wsize0;

        outshapes[0] = convInferShape(net, inpsize, params, wsize, symbolic);
    }

    virtual void forward(Net2& net, Graph& graph,
                        const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs,
                        std::vector<Buffer>& tempbufs) CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        const Tensor& inp = inputs[0];
        const Tensor* residual = nullptr;
        const void* resptr = nullptr;
        int inptype = inp.type();
        TensorSize inpsize = inp.size();
        CV_Assert(inpsize.layout == LAYOUT_NCHWc);
        CV_Assert(inp.isContinuous());

        if (add_residual) {
            residual = &inputs[ninputs-1];
            resptr = residual->data();
            ninputs--;
        }

        bool dynamic_weights = ninputs > 1;
        if (dynamic_weights) {
            setWeights(inputs[1], ninputs > 2 ? inputs[2] : Tensor(),
                       inpsize.size[inpsize.ndims-1], net.getAccuracy());
        }

        TensorSize outsize = convInferShape(net, inpsize, params, wsize0);
        int outtype = inferType(inptype);
        outputs.resize(1);
        Tensor& out = outputs[0];
        out.fitSameDevice(inp, outsize, outtype);
        CV_Assert(out.isContinuous());

        if (add_residual) {
            CV_Assert(outsize == residual->size());
            CV_Assert(outtype == residual->type());
        }

        CV_Assert(inpsize.layout == LAYOUT_NCHWc);
        int nspatialdims = inpsize.ndims - 3;
        CV_Assert(wsize0.ndims == nspatialdims+2);

        if (inp.empty())
            return;

        const void* inptr = inp.data();
        void* outptr = out.data();
        const void* wptr = weights.data();

        int64_t ksize = 1;
        for (int i = 0; i < nspatialdims; i++)
            ksize *= wsize0.size[wsize0.ndims - nspatialdims + i];
        AutoBuffer<int64_t> buf(ksize*2);
        int64_t* ofstab = buf.data();
        int* yxtab = (int*)(ofstab + ksize);

        ConvState cs = initConvState(net, inpsize, wsize0, params, activ, yxtab, ofstab);
        bool conv1x1 = cs.Hk == 1 && cs.Wk == 1;
        bool depthwise = cs.ngroups == cs.C;
        const float* bias_data = bias.ptr<float>();

        if (batchNorm) {
            fuseBatchNormWeights();
            bias_data = fused_bias.ptr<float>();
        }

        if (depthwise) {
            depthwise_conv2d_t func = getDepthwiseConv2DFunc(inptype);
            CV_Assert(func != nullptr);

            func(inptr, resptr, outptr, cs, wptr,
                 fused_scale.ptr<float>(), bias_data);
        } else {
            if (!conv1x1 && (ofs0.empty() || !cs.sameShape(prev_cs))) {
                initConv2DTables(cs, ofsbuf, ofs0, ofsptrs, mask);
                prev_cs = cs;
            }

            conv_func_t func = conv1x1 ?
                (inptype == CV_32F ? conv2d_1x1_32f : nullptr) :
                (inptype == CV_32F ? conv2d_32f : nullptr);
            CV_Assert(func != nullptr);

            func(inptr, resptr, outptr, cs, wptr,
                 fused_scale.ptr<float>(), bias_data, ofs0.data(),
                 (const int32_t**)ofsptrs.data(), mask.data());
        }

        if (dynamic_weights) {
            // to keep memory footprint low in the case of
            // very rare situation of dynamic convolution weights,
            // we release temporarily allocated and reordered copy of the weights
            weights.release();
        }
    }

    Tensor weights, bias, fused_scale, fused_bias;
    TensorSize wsize0;
    ConvState prev_cs;
    std::vector<int32_t> ofsbuf;
    std::vector<int32_t> ofs0;
    std::vector<int32_t*> ofsptrs;
    std::vector<uint8_t> mask;
};

ConvOp::~ConvOp() {}

Op ConvOp::create(const ConvParams& params)
{
    return std::make_shared<ConvOpImpl>(params);
}

Arg conv(Graph& graph, std::string_view opname, std::string_view outname,
         Arg input, Arg weights, Arg bias, const ConvParams& params)
{
    Op op = ConvOp::create(params);
    std::vector<Arg> inputs = {input, weights};
    if (!bias.empty())
        inputs.push_back(bias);
    
    return graph->append(opname, op, outname, inputs);
}

}}

