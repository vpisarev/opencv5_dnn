// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv { namespace dnn {

static void maxpool2d_32f(const void* inp_, void* out_, const DepthwiseConvParams& dwparams)
{
    int nlanes_ = VTraits<v_float32>::vlanes();
    int C0_ = (int)dwparams.C0;

    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);

    int64_t NC = dwparams.N*dwparams.C1;
    int ksize_ = (int)(dwparams.KH*dwparams.KW);
    AutoBuffer<int64_t> local_ofstab_buf(ksize_);
    int64_t* local_ofstab = local_ofstab_buf.data();
    for (int k = 0; k < ksize_; k++)
        local_ofstab[k] = dwparams.ofstab[k]*C0_;

    parallel_for_(Range(0, (int)NC), [&](const Range& r) {
        int64_t nc0 = r.start, nc1 = r.end;
        int nlanes = nlanes_, C0 = C0_;
        int64_t Hi = dwparams.Hi, Wi = dwparams.Wi;
        int64_t H = dwparams.H, W = dwparams.W;
        int64_t iplanesize = Hi*Wi*C0;
        int64_t planesize = H*W*C0;
        int64_t SY = dwparams.SY, SX = dwparams.SX;
        int64_t DY = dwparams.DY, DX = dwparams.DX;
        int64_t pad_y0 = dwparams.pad_y0, pad_x0 = dwparams.pad_x0;
        int64_t pad_y1 = dwparams.pad_y1, pad_x1 = dwparams.pad_x1;
        int64_t inner_y0 = dwparams.inner_y0, inner_y1 = dwparams.inner_y1;
        int64_t inner_x0 = dwparams.inner_x0, inner_x1 = dwparams.inner_x1;
        int ksize = ksize_;
        const int* yxtab = dwparams.yxtab;
        const int64_t* ofstab = local_ofstab;

        const float* inp = (const float*)inp_ + nc0*iplanesize;
        float* out = (float*)out_ + nc0*planesize;
        v_float32 s_min = vx_setall_f32(-FLT_MAX);

        for (int64_t nc = nc0; nc < nc1; nc++, inp += iplanesize) {
            for (int64_t y0 = 0; y0 < H; y0++, out += W*C0) {
                int64_t x0 = 0, x1 = y0 >= inner_y0 && y0 < inner_y1 ? inner_x0 : W;
                int64_t yi_ = y0*SY - pad_y0;
                for(;;) {
                    if (nlanes == C0) {
                        for (; x0 < x1; x0++) {
                            int64_t xi_ = x0*SX - pad_x0;
                            v_float32 s0 = s_min;
                            for (int k = 0; k < ksize; k++) {
                                int64_t yi = yi_ + yxtab[k*2];
                                int64_t xi = xi_ + yxtab[k*2+1];
                                v_float32 v0;
                                if ((uint64_t)yi >= (uint64_t)Hi || (uint64_t)xi >= (uint64_t)Wi)
                                    continue;
                                v0 = vx_load(inp + (yi*Wi + xi)*C0);
                                s0 = v_max(s0, v0);
                            }
                            vx_store(out + x0*C0, s0);
                        }
                    } else {
                        for (; x0 < x1; x0++) {
                            int64_t xi_ = x0*SX - pad_x0;
                            for (int c = 0; c < C0; c += nlanes*2) {
                                v_float32 s0 = s_min, s1 = s_min;
                                for (int k = 0; k < ksize; k++) {
                                    int64_t yi = yi_ + yxtab[k*2];
                                    int64_t xi = xi_ + yxtab[k*2+1];
                                    v_float32 v0, v1;
                                    if ((uint64_t)yi >= (uint64_t)Hi || (uint64_t)xi >= (uint64_t)Wi)
                                        continue;
                                    int64_t ofs_k = (yi*Wi + xi)*C0 + c;
                                    v0 = vx_load(inp + ofs_k);
                                    v1 = vx_load(inp + ofs_k + nlanes);
                                    s0 = v_max(s0, v0);
                                    s1 = v_max(s1, v1);
                                }
                                vx_store(out + x0*C0 + c, s0);
                                vx_store(out + x0*C0 + c + nlanes, s1);
                            }
                        }
                    }
                    if (x0 == W)
                        break;
                    x1 = inner_x1;
                    if (nlanes == C0) {
                        for (; x0 < x1; x0++) {
                            int64_t xi_ = x0*SX - pad_x0;
                            const float* inp_xi = inp + (Wi*yi_ + xi_)*C0;

                            v_float32 s0 = vx_load(inp_xi + ofstab[0]);
                            for (int k = 1; k < ksize; k++)
                                s0 = v_max(s0, vx_load(inp_xi + ofstab[k]));
                            vx_store(out + x0*C0, s0);
                        }
                    } else if (nlanes*2 == C0) {
                        for (; x0 < x1; x0++) {
                            int64_t xi_ = x0*SX - pad_x0;
                            const float* inp_xi = inp + (Wi*yi_ + xi_)*C0;

                            int64_t ofs_k = ofstab[0];
                            v_float32 s0 = vx_load(inp_xi + ofs_k);
                            v_float32 s1 = vx_load(inp_xi + ofs_k + nlanes);
                            for (int k = 1; k < ksize; k++) {
                                ofs_k = ofstab[k];
                                s0 = v_max(s0, vx_load(inp_xi + ofs_k));
                                s1 = v_max(s1, vx_load(inp_xi + ofs_k + nlanes));
                            }
                            vx_store(out + x0*C0, s0);
                            vx_store(out + x0*C0 + nlanes, s1);
                        }
                    } else {
                        for (; x0 < x1; x0++) {
                            int64_t xi_ = x0*SX - pad_x0;
                            for (int c = 0; c < C0; c += nlanes*4) {
                                const float* inp_xi = inp + (Wi*yi_ + xi_)*C0 + c;

                                int64_t ofs_k = ofstab[0];
                                v_float32 s0 = vx_load(inp_xi + ofs_k);
                                v_float32 s1 = vx_load(inp_xi + ofs_k + nlanes);
                                v_float32 s2 = vx_load(inp_xi + ofs_k + nlanes*2);
                                v_float32 s3 = vx_load(inp_xi + ofs_k + nlanes*3);
                                for (int k = 1; k < ksize; k++) {
                                    ofs_k = ofstab[k];
                                    s0 = v_max(s0, vx_load(inp_xi + ofs_k));
                                    s1 = v_max(s1, vx_load(inp_xi + ofs_k + nlanes));
                                    s2 = v_max(s2, vx_load(inp_xi + ofs_k + nlanes*2));
                                    s3 = v_max(s3, vx_load(inp_xi + ofs_k + nlanes*3));
                                }
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
        }
    });
}

template<typename _Tp>
static void maxpool2d_16(const _Tp* inp_, _Tp* out_, const DepthwiseConvParams& dwparams)
{
    int nlanes_ = VTraits<v_float32>::vlanes();
    int C0_ = (int)dwparams.C0;

    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);

    int64_t NC = dwparams.N*dwparams.C1;
    int ksize_ = (int)(dwparams.KH*dwparams.KW);
    AutoBuffer<int64_t> local_ofstab_buf(ksize_);
    int64_t* local_ofstab = local_ofstab_buf.data();
    for (int k = 0; k < ksize_; k++)
        local_ofstab[k] = dwparams.ofstab[k]*C0_;

    parallel_for_(Range(0, (int)NC), [&](const Range& r) {
        int64_t nc0 = r.start, nc1 = r.end;
        int nlanes = nlanes_, C0 = C0_;
        int64_t Hi = dwparams.Hi, Wi = dwparams.Wi;
        int64_t H = dwparams.H, W = dwparams.W;
        int64_t iplanesize = Hi*Wi*C0;
        int64_t planesize = H*W*C0;
        int64_t SY = dwparams.SY, SX = dwparams.SX;
        int64_t DY = dwparams.DY, DX = dwparams.DX;
        int64_t pad_y0 = dwparams.pad_y0, pad_x0 = dwparams.pad_x0;
        int64_t pad_y1 = dwparams.pad_y1, pad_x1 = dwparams.pad_x1;
        int64_t inner_y0 = dwparams.inner_y0, inner_y1 = dwparams.inner_y1;
        int64_t inner_x0 = dwparams.inner_x0, inner_x1 = dwparams.inner_x1;
        int ksize = ksize_;
        const int* yxtab = dwparams.yxtab;
        const int64_t* ofstab = local_ofstab;

        const _Tp* inp = inp_ + nc0*iplanesize;
        _Tp* out = out_ + nc0*planesize;
        v_float32 s_min = vx_setall_f32(-FLT_MAX);

        for (int64_t nc = nc0; nc < nc1; nc++, inp += iplanesize) {
            for (int64_t y0 = 0; y0 < H; y0++, out += W*C0) {
                int64_t x0 = 0, x1 = y0 >= inner_y0 && y0 < inner_y1 ? inner_x0 : W;
                int64_t yi_ = y0*SY - pad_y0;
                for(;;) {
                    if (nlanes == C0) {
                        for (; x0 < x1; x0++) {
                            int64_t xi_ = x0*SX - pad_x0;
                            v_float32 s0 = s_min;
                            for (int k = 0; k < ksize; k++) {
                                int64_t yi = yi_ + yxtab[k*2];
                                int64_t xi = xi_ + yxtab[k*2+1];
                                v_float32 v0;
                                if ((uint64_t)yi >= (uint64_t)Hi || (uint64_t)xi >= (uint64_t)Wi)
                                    continue;
                                v0 = vx_load_expand(inp + (yi*Wi + xi)*C0);
                                s0 = v_max(s0, v0);
                            }
                            v_pack_store(out + x0*C0, s0);
                        }
                    } else {
                        for (; x0 < x1; x0++) {
                            int64_t xi_ = x0*SX - pad_x0;
                            for (int c = 0; c < C0; c += nlanes*2) {
                                v_float32 s0 = s_min, s1 = s_min;
                                for (int k = 0; k < ksize; k++) {
                                    int64_t yi = yi_ + yxtab[k*2];
                                    int64_t xi = xi_ + yxtab[k*2+1];
                                    v_float32 v0, v1;
                                    if ((uint64_t)yi >= (uint64_t)Hi || (uint64_t)xi >= (uint64_t)Wi)
                                        continue;
                                    int64_t ofs_k = (yi*Wi + xi)*C0 + c;
                                    v0 = vx_load_expand(inp + ofs_k);
                                    v1 = vx_load_expand(inp + ofs_k + nlanes);
                                    s0 = v_max(s0, v0);
                                    s1 = v_max(s1, v1);
                                }
                                v_pack_store(out + x0*C0 + c, s0);
                                v_pack_store(out + x0*C0 + c + nlanes, s1);
                            }
                        }
                    }
                    if (x0 == W)
                        break;
                    x1 = inner_x1;
                    if (nlanes == C0) {
                        for (; x0 < x1; x0++) {
                            int64_t xi_ = x0*SX - pad_x0;
                            const _Tp* inp_xi = inp + (Wi*yi_ + xi_)*C0;

                            v_float32 s0 = vx_load_expand(inp_xi + ofstab[0]);
                            for (int k = 1; k < ksize; k++)
                                s0 = v_max(s0, vx_load_expand(inp_xi + ofstab[k]));
                            v_pack_store(out + x0*C0, s0);
                        }
                    } else if (nlanes*2 == C0) {
                        for (; x0 < x1; x0++) {
                            int64_t xi_ = x0*SX - pad_x0;
                            const _Tp* inp_xi = inp + (Wi*yi_ + xi_)*C0;

                            int64_t ofs_k = ofstab[0];
                            v_float32 s0 = vx_load_expand(inp_xi + ofs_k);
                            v_float32 s1 = vx_load_expand(inp_xi + ofs_k + nlanes);
                            for (int k = 1; k < ksize; k++) {
                                ofs_k = ofstab[k];
                                s0 = v_max(s0, vx_load_expand(inp_xi + ofs_k));
                                s1 = v_max(s1, vx_load_expand(inp_xi + ofs_k + nlanes));
                            }
                            v_pack_store(out + x0*C0, s0);
                            v_pack_store(out + x0*C0 + nlanes, s1);
                        }
                    } else {
                        for (; x0 < x1; x0++) {
                            int64_t xi_ = x0*SX - pad_x0;
                            for (int c = 0; c < C0; c += nlanes*4) {
                                const _Tp* inp_xi = inp + (Wi*yi_ + xi_)*C0 + c;

                                int64_t ofs_k = ofstab[0];
                                v_float32 s0 = vx_load_expand(inp_xi + ofs_k);
                                v_float32 s1 = vx_load_expand(inp_xi + ofs_k + nlanes);
                                v_float32 s2 = vx_load_expand(inp_xi + ofs_k + nlanes*2);
                                v_float32 s3 = vx_load_expand(inp_xi + ofs_k + nlanes*3);
                                for (int k = 1; k < ksize; k++) {
                                    ofs_k = ofstab[k];
                                    s0 = v_max(s0, vx_load_expand(inp_xi + ofs_k));
                                    s1 = v_max(s1, vx_load_expand(inp_xi + ofs_k + nlanes));
                                    s2 = v_max(s2, vx_load_expand(inp_xi + ofs_k + nlanes*2));
                                    s3 = v_max(s3, vx_load_expand(inp_xi + ofs_k + nlanes*3));
                                }
                                v_pack_store(out + x0*C0 + c, s0);
                                v_pack_store(out + x0*C0 + c + nlanes, s1);
                                v_pack_store(out + x0*C0 + c + nlanes*2, s2);
                                v_pack_store(out + x0*C0 + c + nlanes*3, s3);
                            }
                        }
                    }
                    x1 = W;
                }
            }
        }
    });
}

static void maxpool2d_16f(const void* inp_, void* out_, const DepthwiseConvParams& dwparams)
{
    maxpool2d_16((const float16_t*)inp_, (float16_t*)out_, dwparams);
}

static void maxpool2d_16bf(const void* inp_, void* out_, const DepthwiseConvParams& dwparams)
{
    maxpool2d_16((const bfloat16_t*)inp_, (bfloat16_t*)out_, dwparams);
}

typedef void (*maxpool_func_t)(const void* inp, void* out, const DepthwiseConvParams& dwparams);

class MaxPoolOpImpl : public MaxPoolOp
{
public:
    MaxPoolOpImpl(const ConvParams& convparams_, bool computeIndices_, bool rowMajorOrder_)
    {
        params = convparams_;
        computeIndices = computeIndices_;
        rowMajorOrder = rowMajorOrder_;

        // currently we don't support it
        CV_Assert(!computeIndices);
    }
    virtual std::string_view name() const CV_OVERRIDE { return "MaxPool"; }
    virtual Op clone() const CV_OVERRIDE
    {
        return std::make_shared<MaxPoolOpImpl>(params, computeIndices, rowMajorOrder);
    }

    virtual int minNumInputs() const CV_OVERRIDE { return 1; }
    virtual int maxNumInputs() const CV_OVERRIDE { return 1; }
    virtual int minNumOutputs() const CV_OVERRIDE { return 1; }
    virtual int maxNumOutputs() const CV_OVERRIDE { return 1; }

    int inferType(int inptype0) const
    {
        return inptype0;
    }

    virtual bool supportType(int, int depth) const CV_OVERRIDE
    {
        return depth == CV_32F || depth == CV_16F || depth == CV_16BF;
    }

    virtual bool alwaysSupportInplace() const CV_OVERRIDE
    {
        return false;
    }

    virtual int64_t getFLOPS(const std::vector<SizeType> &inputs,
                             const std::vector<SizeType> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        CV_Assert(outputs.size() == 1);
        // probably, there should be a coefficient in the case of complex reduction functions
        int64_t ksize = 1;
        for (auto sz: params.ksizes) ksize *= sz;
        return (int64_t)(inputs[0].size.total()*ksize);
    }

    virtual void inferShapes(const Net2& net, const Graph& graph,
                            const std::vector<Arg>& inpargs,
                            const std::vector<SizeType>& inpst,
                            const std::vector<Arg>& outargs,
                            std::vector<SizeType>& outst,
                            std::vector<size_t>& tempbufs) const CV_OVERRIDE
    {
        int ninputs = (int)inpargs.size(), noutputs = (int)outargs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        CV_Assert(noutputs == 1);
        outst.resize(1);

        const TensorSize& inpsize = inpst[0].size;
        TensorSize& outsize = outst[0].size;

        outsize = convInferShape(inpsize, params);
        outst[0].type = inferType(inpst[0].type);
        tempbufs.assign(1, (size_t)0);
    }

    virtual void forward(Net2& net, Graph& graph,
                        const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs,
                        std::vector<Buffer>& tempbufs) CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        const Tensor& inp = inputs[0];
        CV_Assert(inp.isContinuous());

        int inptype = inp.type(), outtype = inferType(inptype);
        TensorSize inpsize = inp.size();
        TensorSize outsize = convInferShape(inpsize, params);
        outputs.resize(1);
        Tensor& out = outputs[0];
        out.fitSameDevice(inp, outsize, outtype);
        CV_Assert(out.isContinuous());

        if (inp.empty())
            return;

        const char* inptr0 = (const char*)inp.data();
        char* outptr0 = (char*)out.data();

        maxpool_func_t func =
            inptype == CV_32F ? maxpool2d_32f :
            inptype == CV_16F ? maxpool2d_16f :
            inptype == CV_16BF ? maxpool2d_16bf : nullptr;

        CV_Assert(func != nullptr);

        int64_t ksize = 1;
        for (auto sz: params.ksizes) ksize *= sz;
        AutoBuffer<int64_t> buf(ksize*2);
        int64_t* ofstab = buf.data();
        int* yxtab = (int*)(ofstab + ksize);

        DepthwiseConvParams dwparams = initDepthwiseConv(inpsize, params, yxtab, ofstab);

        func(inptr0, outptr0, dwparams);
    }
};

MaxPoolOp::~MaxPoolOp() {}

Op MaxPoolOp::create(const ConvParams& params, bool computeIndices, bool rowMajorOrder)
{
    return std::make_shared<MaxPoolOpImpl>(params, computeIndices, rowMajorOrder);
}

}}
