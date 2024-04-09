// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"
#include <math.h>

namespace cv { namespace dnn {

template <typename _Tp>
void batchnorm(const _Tp* inp_, _Tp* out_, const TensorSize& size,
               const float* scale_, const float* bias_)
{
    CV_Assert(size.layout == LAYOUT_NCHWc);

    constexpr int64_t MAX_C0 = 128;
    int64_t N = size.size[0], spatial_size_ = 1;
    int64_t C1 = size.size[1], C0_ = size.size[size.ndims-1];
    for (int k = 2; k < size.ndims-1; k++)
        spatial_size_ *= size.size[k];

    CV_Assert(C0_ <= MAX_C0);

    parallel_for_(Range(0, (int)(N*C1)), [&](const Range& r) {
        int64_t C = size.C, C0 = C0_;
        int64_t spatial_size = spatial_size_;
        float scale[MAX_C0], bias[MAX_C0];

        for (int64_t nc = r.start; nc < r.end; nc++) {
            int64_t n = nc/C1;
            int64_t c1 = nc - n*C1;
            int64_t c, c0 = std::min(C0, C - c1*C0);

            for (c = 0; c < c0; c++) {
                scale[c] = scale_[c1*C0 + c];
                bias[c] = bias_[c1*C0 + c];
            }
            
            for (; c < C0; c++)
                scale[c] = bias[c] = 0.f;

            const _Tp* inp = inp_ + nc*spatial_size*C0;
            _Tp* out = out_ + nc*spatial_size*C0;

            for (int64_t k = 0; k < spatial_size*C0; k += C0) {
                for (int64_t c = 0; c < C0; c++)
                    out[k*C0 + c] = _Tp((float)inp[k*C0 + c]*scale[c] + bias[c]);
            }
        }
    });
}

#undef CV_BATCHNORM_IMPL
#define CV_BATCHNORM_IMPL(typ, suffix) \
static void batchnorm_##suffix(const void* inp_, void* out_, const TensorSize& size, \
                               const float* scale, const float* bias) \
{ \
    batchnorm((const typ*)inp_, (typ*)out_, size, scale, bias); \
}

CV_BATCHNORM_IMPL(hfloat, 16f)
CV_BATCHNORM_IMPL(float, 32f)

typedef void (*batchnorm_func_t)(const void* inp, void* out, const TensorSize& size,
                                 const float* scale, const float* bias);

class BatchNormOpImpl : public BatchNormOp
{
public:
    BatchNormOpImpl(double epsilon_)
    {
        epsilon = epsilon_;
    }
    virtual std::string_view name() const CV_OVERRIDE { return "BatchNormalization"; }
    virtual Op clone() const CV_OVERRIDE
    {
        return std::make_shared<BatchNormOpImpl>(epsilon);
    }

    virtual int minNumInputs() const CV_OVERRIDE { return 1; }
    virtual int maxNumInputs() const CV_OVERRIDE { return 5; }
    virtual int minNumOutputs() const CV_OVERRIDE { return 1; }
    virtual int maxNumOutputs() const CV_OVERRIDE { return 1; }

    int inferType(int inptype0) const
    {
        return inptype0;
    }

    virtual bool supportType(int, int type) const CV_OVERRIDE
    {
        return type == CV_32F || type == CV_16F;
    }

    virtual bool alwaysSupportInplace() const CV_OVERRIDE
    {
        return true;
    }

    virtual int64_t getFLOPS(const std::vector<SizeType> &inputs,
                             const std::vector<SizeType> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        return (int64_t)inputs[0].size.total();
    }

    TensorSize inferShapes_(const TensorSize& inpsize) const
    {
        int ndims = inpsize.ndims;

        CV_Assert(inpsize.layout != LAYOUT_NCHWc);
        CV_Assert(ndims >= 4);

        TensorSize outsize = inpsize;
        return outsize;
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
        outst[0].size = inferShapes_(inpsize);
        outst[0].type = inferType(inpst[0].type);
        tempbufs.assign(1, (size_t)0);
    }

    virtual void computeScaleBias(const Tensor& scale_, const Tensor& bias_,
                                  const Tensor& mean_, const Tensor& variance_) override
    {
        CV_Assert(scale_.isContinuous() && bias_.isContinuous() &&
                  mean_.isContinuous() && variance_.isContinuous());
        int scale_type = scale_.type(), bias_type = bias_.type();
        int mean_type = mean_.type(), var_type = variance_.type();
        CV_Assert((scale_type == CV_32F || scale_type == CV_16F) &&
                  (bias_type == CV_32F || bias_type == CV_16F) &&
                  (mean_type == CV_32F || mean_type == CV_16F) &&
                  (var_type == CV_32F || var_type == CV_16F));
        int64_t C = (int64_t)scale_.total();
        CV_Assert((int64_t)bias_.total() == C &&
                  (int64_t)mean_.total() == C &&
                  (int64_t)variance_.total() == C);
        scale.fit(TensorSize(1, &C), CV_32F);
        bias.fit(TensorSize(1, &C), CV_32F);
        Tensor scale_host = scale_.download(), bias_host = bias_.download();
        Tensor mean_host = mean_.download(), var_host = variance_.download();

        const void* scale0_data = scale_host.data();
        const void* bias0_data = bias_host.data();
        const void* mean_data = mean_host.data();
        const void* var_data = var_host.data();

        float* scale_data = scale.ptr<float>();
        float* bias_data = bias.ptr<float>();

        for (int64_t i = 0; i < C; i++) {
            float s0 = scale_type == CV_32F ?
                ((const float*)scale0_data)[i] : (float)((const hfloat*)scale0_data)[i];
            float b0 = bias_type == CV_32F ?
                ((const float*)bias0_data)[i] : (float)((const hfloat*)bias0_data)[i];
            float m = mean_type == CV_32F ?
                ((const float*)mean_data)[i] : (float)((const hfloat*)mean_data)[i];
            float v = var_type == CV_32F ?
                ((const float*)var_data)[i] : (float)((const hfloat*)var_data)[i];
            scale_data[i] = s0/sqrtf(v + epsilon);
            bias_data[i] = b0 - m*scale_data[i];
        }
    }

    virtual void forward(Net2& net, Graph& graph,
                        const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs,
                        std::vector<Buffer>& tempbufs) CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(ninputs == 1 || ninputs == 5);
        const Tensor& inp = inputs[0];
        CV_Assert(inp.isContinuous());

        if (ninputs > 1) {
            computeScaleBias(inputs[1], inputs[2], inputs[3], inputs[4]);
        }

        int inptype = inp.type(), outtype = inferType(inptype);
        TensorSize inpsize = inp.size();
        TensorSize outsize = inferShapes_(inpsize);
        outputs.resize(1);
        Tensor& out = outputs[0];
        out.fitSameDevice(inp, outsize, outtype);

        batchnorm_func_t func =
            inptype == CV_16F ? batchnorm_16f :
            inptype == CV_32F ? batchnorm_32f : nullptr;

        CV_Assert(func != nullptr);

        func(inp.data(), out.data(), inpsize, scale.ptr<float>(), bias.ptr<float>());
    }
};

BatchNormOp::~BatchNormOp() {}

Op BatchNormOp::create(double epsilon_)
{
    return std::make_shared<BatchNormOpImpl>(epsilon_);
}

}}
