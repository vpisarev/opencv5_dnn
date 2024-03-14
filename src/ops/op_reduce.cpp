// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"
#include <math.h>

namespace cv { namespace dnn {

#define CV_REDUCE_OP_MIN(a, b) std::min((a), (b))
#define CV_REDUCE_OP_MAX(a, b) std::max((a), (b))
#define CV_REDUCE_OP_SUM(a, b) ((a) + (b))
#define CV_REDUCE_OP_SUM_ABSF(a, b) ((a) + std::abs(b))
#define CV_REDUCE_OP_SUM_ABS(a, b) ((a) + std::abs(b))
#define CV_REDUCE_OP_SUM_SQR(a, b) ((a) + (b)*(b))
#define CV_REDUCE_OP_PROD(a, b) ((a) * (b))

#define CV_REDUCE_IMPL(typ, acctyp, suffix, op, val0) \
static void reduce_##suffix(const void* inptr_, int64_t ystep, void* accptr_, \
                            int64_t nrows, int64_t ncols, int rdim, bool init) { \
    const typ* inptr = (const typ*)inptr_; \
    acctyp* accptr = (acctyp*)accptr_; \
    if (init) { \
        int64_t nout = rdim != 0 ? nrows : ncols; \
        for (int64_t j = 0; j < nout; j++) \
            accptr[j] = val0; \
    } \
    if (rdim != 0) { \
        for (int64_t i = 0; i < nrows; i++) { \
            acctyp acc = accptr[i]; \
            for (int64_t j = 0; j < ncols; j++) { \
                acctyp x = (acctyp)inptr[i*ystep + j]; \
                acc = op(acc, x); \
            } \
            accptr[i] = acc; \
        } \
    } else { \
        for (int64_t i = 0; i < nrows; i++) { \
            for (int64_t j = 0; j < ncols; j++) { \
                acctyp acc = accptr[j]; \
                acctyp x = (acctyp)inptr[i*ystep + j]; \
                accptr[j] = op(acc, x); \
            } \
        } \
    } \
}

CV_REDUCE_IMPL(int8_t, int8_t, max_8s, CV_REDUCE_OP_MAX, -128)
CV_REDUCE_IMPL(int8_t, int8_t, min_8s, CV_REDUCE_OP_MIN, 127)
CV_REDUCE_IMPL(uint8_t, uint8_t, max_8u, CV_REDUCE_OP_MAX, 0)
CV_REDUCE_IMPL(uint8_t, uint8_t, min_8u, CV_REDUCE_OP_MIN, 255)
CV_REDUCE_IMPL(int32_t, int32_t, max_32s, CV_REDUCE_OP_MAX, INT_MIN)
CV_REDUCE_IMPL(int32_t, int32_t, min_32s, CV_REDUCE_OP_MIN, INT_MAX)
CV_REDUCE_IMPL(uint32_t, uint32_t, max_32u, CV_REDUCE_OP_MAX, 0)
CV_REDUCE_IMPL(uint32_t, uint32_t, min_32u, CV_REDUCE_OP_MIN, UINT_MAX)
CV_REDUCE_IMPL(int64_t, int64_t, max_64s, CV_REDUCE_OP_MAX, 0x8000000000000000LL)
CV_REDUCE_IMPL(int64_t, int64_t, min_64s, CV_REDUCE_OP_MIN, 0x7fffffffffffffffLL)
CV_REDUCE_IMPL(uint64_t, uint64_t, max_64u, CV_REDUCE_OP_MAX, 0)
CV_REDUCE_IMPL(uint64_t, uint64_t, min_64u, CV_REDUCE_OP_MIN, 0xffffffffffffffffUL)
CV_REDUCE_IMPL(float, float, max_32f, CV_REDUCE_OP_MAX, -FLT_MAX)
CV_REDUCE_IMPL(float, float, min_32f, CV_REDUCE_OP_MIN, FLT_MAX)
CV_REDUCE_IMPL(cv::float16_t, float, max_16f32f, CV_REDUCE_OP_MAX, -FLT_MAX)
CV_REDUCE_IMPL(cv::float16_t, float, min_16f32f, CV_REDUCE_OP_MIN, FLT_MAX)
CV_REDUCE_IMPL(cv::bfloat16_t, float, max_16bf32f, CV_REDUCE_OP_MAX, -FLT_MAX)
CV_REDUCE_IMPL(cv::bfloat16_t, float, min_16bf32f, CV_REDUCE_OP_MIN, FLT_MAX)

CV_REDUCE_IMPL(int8_t, int64_t, sum_8s64s, CV_REDUCE_OP_SUM, 0)
CV_REDUCE_IMPL(uint8_t, uint64_t, sum_8u64u, CV_REDUCE_OP_SUM, 0)
CV_REDUCE_IMPL(int32_t, int64_t, sum_32s64s, CV_REDUCE_OP_SUM, 0)
CV_REDUCE_IMPL(uint32_t, uint64_t, sum_32u64u, CV_REDUCE_OP_SUM, 0)
CV_REDUCE_IMPL(int64_t, int64_t, sum_64s, CV_REDUCE_OP_SUM, 0)
CV_REDUCE_IMPL(uint64_t, uint64_t, sum_64u, CV_REDUCE_OP_SUM, 0)
CV_REDUCE_IMPL(float, double, sum_32f64f, CV_REDUCE_OP_SUM, 0.)
CV_REDUCE_IMPL(cv::float16_t, float, sum_16f32f, CV_REDUCE_OP_SUM, 0.f)
CV_REDUCE_IMPL(cv::bfloat16_t, float, sum_16bf32f, CV_REDUCE_OP_SUM, 0.f)

CV_REDUCE_IMPL(int32_t, int64_t, sum_abs_32s64s, CV_REDUCE_OP_SUM_ABS, 0)
CV_REDUCE_IMPL(int64_t, int64_t, sum_abs_64s, CV_REDUCE_OP_SUM_ABS, 0)
CV_REDUCE_IMPL(float, double, sum_abs_32f64f, CV_REDUCE_OP_SUM_ABSF, 0.)
CV_REDUCE_IMPL(cv::float16_t, float, sum_abs_16f32f, CV_REDUCE_OP_SUM_ABSF, 0.f)
CV_REDUCE_IMPL(cv::bfloat16_t, float, sum_abs_16bf32f, CV_REDUCE_OP_SUM_ABSF, 0.f)

CV_REDUCE_IMPL(int32_t, int64_t, sum_sqr_32s64u, CV_REDUCE_OP_SUM_SQR, 0)
CV_REDUCE_IMPL(uint32_t, uint64_t, sum_sqr_32u64u, CV_REDUCE_OP_SUM_SQR, 0)
CV_REDUCE_IMPL(int64_t, int64_t, sum_sqr_64s, CV_REDUCE_OP_SUM_SQR, 0)
CV_REDUCE_IMPL(uint64_t, uint64_t, sum_sqr_64u, CV_REDUCE_OP_SUM_SQR, 0)
CV_REDUCE_IMPL(float, double, sum_sqr_32f64f, CV_REDUCE_OP_SUM_SQR, 0.)
CV_REDUCE_IMPL(cv::float16_t, double, sum_sqr_16f64f, CV_REDUCE_OP_SUM_SQR, 0.)
CV_REDUCE_IMPL(cv::bfloat16_t, double, sum_sqr_16bf64f, CV_REDUCE_OP_SUM_SQR, 0.)

CV_REDUCE_IMPL(int32_t, int64_t, prod_32s64s, CV_REDUCE_OP_PROD, 1)
CV_REDUCE_IMPL(uint32_t, uint64_t, prod_32u64u, CV_REDUCE_OP_PROD, 1)
CV_REDUCE_IMPL(int64_t, int64_t, prod_64s, CV_REDUCE_OP_PROD, 1)
CV_REDUCE_IMPL(uint64_t, uint64_t, prod_64u, CV_REDUCE_OP_PROD, 1)
CV_REDUCE_IMPL(float, double, prod_32f64f, CV_REDUCE_OP_PROD, 1.)
CV_REDUCE_IMPL(cv::float16_t, double, prod_16f64f, CV_REDUCE_OP_PROD, 1.)
CV_REDUCE_IMPL(cv::bfloat16_t, double, prod_16bf64f, CV_REDUCE_OP_PROD, 1.)

#undef CV_REDUCE_FINIT_COPY
#define CV_REDUCE_FINIT_COPY(typ, suffix) \
static void reduce_finit_copy_##suffix(const void* inptr, void* outptr, int64_t ncols, const double*) \
{ \
    for (int64_t j = 0; j < ncols; j++) \
        ((typ*)outptr)[j] = ((typ*)inptr)[j]; \
}

CV_REDUCE_FINIT_COPY(uint8_t, 8u)
CV_REDUCE_FINIT_COPY(uint16_t, 16u)
CV_REDUCE_FINIT_COPY(uint32_t, 32u)
CV_REDUCE_FINIT_COPY(uint64_t, 64u)

#undef CV_REDUCE_FINIT_CAST
#define CV_REDUCE_FINIT_CAST(inptyp, outtyp, suffix) \
static void reduce_finit_cast_##suffix(const void* inptr, void* outptr, int64_t ncols, const double*) \
{ \
    for (int64_t j = 0; j < ncols; j++) \
        ((outtyp*)outptr)[j] = saturate_cast<outtyp>(((inptyp*)inptr)[j]); \
}

CV_REDUCE_FINIT_CAST(int64_t, int32_t, 64s32s)
CV_REDUCE_FINIT_CAST(uint64_t, int32_t, 64u32s)
CV_REDUCE_FINIT_CAST(uint64_t, uint32_t, 64u32u)
CV_REDUCE_FINIT_CAST(double, float, 64f32f)
CV_REDUCE_FINIT_CAST(float, cv::float16_t, 32f16f)
CV_REDUCE_FINIT_CAST(float, cv::bfloat16_t, 32f16bf)
CV_REDUCE_FINIT_CAST(double, cv::float16_t, 64f16f)
CV_REDUCE_FINIT_CAST(double, cv::bfloat16_t, 64f16bf)

#undef CV_REDUCE_FINIT_SCALE
#define CV_REDUCE_FINIT_SCALE(inptyp, outtyp, suffix) \
static void reduce_finit_scale_##suffix(const void* inptr_, void* outptr_, int64_t ncols, const double* params) \
{ \
    const inptyp* inptr = (const inptyp*)inptr_; \
    outtyp* outptr = (outtyp*)outptr_; \
    inptyp param = (inptyp)params[0]; \
    for (int64_t j = 0; j < ncols; j++) \
        outptr[j] = saturate_cast<outtyp>(inptr[j]*param); \
}

CV_REDUCE_FINIT_SCALE(double, float, 64f32f)
CV_REDUCE_FINIT_SCALE(float, cv::float16_t, 32f16f)
CV_REDUCE_FINIT_SCALE(float, cv::bfloat16_t, 32f16bf)

#undef CV_REDUCE_FINIT_SQRT
#define CV_REDUCE_FINIT_SQRT(inptyp, outtyp, suffix) \
static void reduce_finit_sqrt_##suffix(const void* inptr_, void* outptr_, int64_t ncols, const double*) \
{ \
    const inptyp* inptr = (const inptyp*)inptr_; \
    outtyp* outptr = (outtyp*)outptr_; \
    for (int64_t j = 0; j < ncols; j++) \
        outptr[j] = saturate_cast<outtyp>(::sqrt((double)inptr[j])); \
}

CV_REDUCE_FINIT_SQRT(double, float, 64f32f)
CV_REDUCE_FINIT_SQRT(double, cv::float16_t, 32f16f)
CV_REDUCE_FINIT_SQRT(double, cv::bfloat16_t, 32f16bf)

typedef void (*reduce_func_t)(const void* inptr, int64_t ystep, void* outptr,
                              int64_t nrows, int64_t ncols, int rdim, bool init);
typedef void (*reduce_finit_func_t)(const void* acc, void* outptr, int64_t ncols, const double* params);

std::string_view reduceOpcode2str(ReduceOpcode opcode_)
{
    constexpr int nopcodes = (int)REDUCE_OPCODE_MAX;
    static const char* names[nopcodes];
    static volatile bool initialized = false;

    if (!initialized) {
        names[REDUCE_NONE] = "<none>";
        names[REDUCE_L1] = "ReduceL1";
        names[REDUCE_L2] = "ReduceL2";
        names[REDUCE_LOGSUM] = "ReduceLogSum";
        names[REDUCE_LOGSUMEXP] = "ReduceLogSumExp";
        names[REDUCE_MAX] = "ReduceMax";
        names[REDUCE_MEAN] = "ReduceMean";
        names[REDUCE_MIN] = "ReduceMin";
        names[REDUCE_PROD] = "ReduceProd";
        names[REDUCE_SUM] = "ReduceSum";
        names[REDUCE_SUM_SQUARE] = "ReduceSumSquare";
        initialized = true;
    }
    int opcode = (int)opcode_;
    CV_Assert(0 <= opcode && opcode < (int)REDUCE_OPCODE_MAX);
    return names[opcode];
}

ReduceOp::~ReduceOp() {}

class ReduceOpImpl : public ReduceOp
{
public:
    enum ReduceFinit
    {
        REDUCE_FINIT_COPY = 0,
        REDUCE_FINIT_CAST,
        REDUCE_FINIT_SCALE,
        REDUCE_FINIT_SQRT
    };

    ReduceOpImpl(ReduceOpcode opcode_, bool keepdims_, bool noOpWithEmptyAxes_)
    {
        opcode = opcode_;
        keepdims = keepdims_;
        noOpWithEmptyAxes = noOpWithEmptyAxes_;
        haveFP16 = checkHardwareSupport(CV_CPU_FP16);
    }
    virtual std::string_view name() const CV_OVERRIDE { return reduceOpcode2str(opcode); }
    virtual Op clone() const CV_OVERRIDE
    {
        return std::make_shared<ReduceOpImpl>(opcode, keepdims, noOpWithEmptyAxes);
    }

    virtual int minNumInputs() const CV_OVERRIDE { return 2; }
    virtual int maxNumInputs() const CV_OVERRIDE { return 2; }
    virtual int minNumOutputs() const CV_OVERRIDE { return 1; }
    virtual int maxNumOutputs() const CV_OVERRIDE { return 1; }

    int inferType(int inptype0) const
    {
        CV_Assert(supportType(0, inptype0));
        return inptype0;
    }

    virtual bool supportType(int input_idx, int depth) const CV_OVERRIDE
    {
        if (input_idx == 1)
            return depth == CV_32S || depth == CV_64S;
        if (depth == CV_32F || depth == CV_16BF || (depth == CV_16F && haveFP16))
            return true;
        if (depth == CV_32S || depth == CV_32U || depth == CV_64S || depth == CV_64U)
            return true;
        if ((opcode == REDUCE_MIN || opcode == REDUCE_MAX) && (depth == CV_8U || depth == CV_8S))
            return true;
        return false;
    }

    virtual bool alwaysSupportInplace() const CV_OVERRIDE
    {
        return false;
    }

    virtual int64_t getFLOPS(const std::vector<SizeType> &inputs,
                           const std::vector<SizeType> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);
        // probably, there should be a coefficient in the case of complex reduction functions
        return (int64_t)inputs[0].size.total();
    }

    int inferShapes_(const TensorSize& inpsize, const Tensor& axes_, int* axes,
                     TensorSize& outsize, TensorSize& outsize_kd, bool* reduce_mask) const
    {
        int axistype = axes_.empty() ? CV_64S : axes_.type();
        int inp_ndims = inpsize.ndims;
        CV_Assert(0 <= inp_ndims && inp_ndims <= TensorSize::MAX_DIMS);
        CV_Assert(axistype == CV_32S || axistype == CV_64S);
        CV_Assert(axes_.isContinuous());
        int naxes = (int)axes_.total();
        if (naxes == 0 && !noOpWithEmptyAxes)
            naxes = inpsize.ndims;
        const int* axes32 = (const int*)axes_.data();
        const int64_t* axes64 = (const int64_t*)axes_.data();
        for (int k = 0; k < inp_ndims; k++)
            reduce_mask[k] = false;

        for (int k = 0; k < naxes; k++) {
            int axis = axistype == CV_32S ? (int)axes32[k] : axes64 ? (int)axes64[k] : k;
            axis = normalizeAxis(axis, inp_ndims);
            if (reduce_mask[axis]) {
                CV_Error(Error::StsError, "there are duplicated axes in the axes specification in Reduce op");
            }
            reduce_mask[axis] = true;
            axes[k] = axis;
        }

        int k1 = 0, out_ndims = keepdims ? inp_ndims : inp_ndims - (int)naxes;
        CV_Assert(out_ndims >= 0);
        for (int k = 0; k < inp_ndims; k++) {
            outsize_kd.size[k] = reduce_mask[k] ? 1 : inpsize.size[k];
            if (keepdims)
                outsize.size[k] = outsize_kd.size[k];
            else if (!reduce_mask[k])
                outsize.size[k1++] = inpsize.size[k];
        }
        outsize.ndims = out_ndims;
        return (int)naxes;
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
        TensorSize outsize_kd;
        const Tensor& axes_ = net.argTensor(inpargs[1]);
        bool reduce_mask[TensorSize::MAX_DIMS];
        int axes[TensorSize::MAX_DIMS];
        int naxes = inferShapes_(inpsize, axes_, axes, outsize, outsize_kd, reduce_mask);
        outst[0].type = inferType(inpst[0].type);
        tempbufs.assign(1, (size_t)0);
    }

    void getReduceFunc(int inptype, int outtype, ReduceOpcode opcode_,
                       int& worktype, reduce_func_t& rf, reduce_finit_func_t& ff) const
    {
        ReduceFinit reduce_finit_type = REDUCE_FINIT_CAST;

        rf = nullptr;
        ff = nullptr;
        worktype = inptype;

        CV_Assert(inptype == outtype);

        if (opcode_ == REDUCE_MIN)
            rf =inptype == CV_8U ? reduce_min_8u :
                inptype == CV_8S ? reduce_min_8s :
                inptype == CV_32U ? reduce_min_32u :
                inptype == CV_32S ? reduce_min_32s :
                inptype == CV_64U ? reduce_min_64u :
                inptype == CV_64S ? reduce_min_64s :
                inptype == CV_32F ? reduce_min_32f :
                inptype == CV_16F ? reduce_min_16f32f :
                inptype == CV_16BF ? reduce_min_16bf32f : nullptr;
        else if (opcode == REDUCE_MAX)
            rf =inptype == CV_8U ? reduce_max_8u :
                inptype == CV_8S ? reduce_max_8s :
                inptype == CV_32U ? reduce_max_32u :
                inptype == CV_32S ? reduce_max_32s :
                inptype == CV_64U ? reduce_max_64u :
                inptype == CV_64S ? reduce_max_64s :
                inptype == CV_32F ? reduce_max_32f :
                inptype == CV_16F ? reduce_max_16f32f :
                inptype == CV_16BF ? reduce_max_16bf32f : nullptr;
        else if (opcode == REDUCE_SUM || opcode == REDUCE_MEAN) {
            rf =inptype == CV_8U ? reduce_sum_8u64u :
                inptype == CV_8S ? reduce_sum_8s64s :
                inptype == CV_32U ? reduce_sum_32u64u :
                inptype == CV_32S ? reduce_sum_32s64s :
                inptype == CV_64U ? reduce_sum_64u :
                inptype == CV_64S ? reduce_sum_64s :
                inptype == CV_32F ? reduce_sum_32f64f :
                inptype == CV_16F ? reduce_sum_16f32f :
                inptype == CV_16BF ? reduce_max_16bf32f : nullptr;
            worktype =
                inptype == CV_8U || inptype == CV_32U || inptype == CV_64U ? CV_64U :
                inptype == CV_8S || inptype == CV_32S || inptype == CV_64S ? CV_64S :
                inptype == CV_16F || inptype == CV_16BF ? CV_32F : CV_64F;
            reduce_finit_type = opcode == REDUCE_MEAN ? REDUCE_FINIT_SCALE : REDUCE_FINIT_CAST;
        } else if (opcode == REDUCE_L1) {
            rf =inptype == CV_32U ? reduce_sum_32u64u :
                inptype == CV_32S ? reduce_sum_abs_32s64s :
                inptype == CV_64U ? reduce_sum_64u :
                inptype == CV_64S ? reduce_sum_abs_64s :
                inptype == CV_32F ? reduce_sum_abs_32f64f :
                inptype == CV_16F ? reduce_sum_abs_16f32f :
                inptype == CV_16BF ? reduce_sum_abs_16bf32f : nullptr;
            worktype =
                inptype == CV_32U || inptype == CV_64U ? CV_64U :
                inptype == CV_32S || inptype == CV_64S ? CV_64S :
                inptype == CV_16F || inptype == CV_16BF ? CV_32F : CV_64F;
        } else if (opcode == REDUCE_L2 || opcode == REDUCE_SUM_SQUARE) {
            rf =inptype == CV_32U ? reduce_sum_sqr_32u64u :
                inptype == CV_32S ? reduce_sum_sqr_32s64u :
                inptype == CV_64U ? reduce_sum_sqr_64u :
                inptype == CV_64S ? reduce_sum_sqr_64s :
                inptype == CV_32F ? reduce_sum_sqr_32f64f :
                inptype == CV_16F ? reduce_sum_sqr_16f64f :
                inptype == CV_16BF ? reduce_sum_sqr_16bf64f : nullptr;
            worktype =
                inptype == CV_32S || inptype == CV_32U || inptype == CV_64S || inptype == CV_64U ? CV_64U : CV_64F;
            reduce_finit_type = opcode == REDUCE_L2 ? REDUCE_FINIT_SQRT : REDUCE_FINIT_CAST;
        } else if (opcode == REDUCE_PROD) {
            rf =inptype == CV_32U ? reduce_prod_32u64u :
                inptype == CV_32S ? reduce_prod_32s64s :
                inptype == CV_64U ? reduce_prod_64u :
                inptype == CV_64S ? reduce_prod_64s :
                inptype == CV_32F ? reduce_prod_32f64f :
                inptype == CV_16F ? reduce_prod_16f64f :
                inptype == CV_16BF ? reduce_prod_16bf64f : nullptr;
            worktype =
                inptype == CV_32S || inptype == CV_64S ? CV_64S :
                inptype == CV_32U || inptype == CV_64U ? CV_64U : CV_64F;
        }

        if (reduce_finit_type == REDUCE_FINIT_CAST) {
            if (worktype == outtype) {
                size_t esz = CV_ELEM_SIZE(worktype);
                ff = esz == 1 ? reduce_finit_copy_8u :
                    esz == 2 ? reduce_finit_copy_16u :
                    esz == 4 ? reduce_finit_copy_32u :
                    esz == 8 ? reduce_finit_copy_64u : nullptr;
            } else {
                ff = outtype == CV_32S && worktype == CV_64S ? reduce_finit_cast_64s32s :
                    outtype == CV_32S && worktype == CV_64U ? reduce_finit_cast_64u32s :
                    outtype == CV_32U && worktype == CV_64U ? reduce_finit_cast_64u32u :
                    outtype == CV_32F && worktype == CV_64F ? reduce_finit_cast_64f32f :
                    outtype == CV_16F && worktype == CV_32F ? reduce_finit_cast_32f16f :
                    outtype == CV_16BF && worktype == CV_32F ? reduce_finit_cast_32f16f :
                    outtype == CV_16F && worktype == CV_64F ? reduce_finit_cast_64f16f :
                    outtype == CV_16BF && worktype == CV_64F ? reduce_finit_cast_64f16f : nullptr;
            }
        } else if (reduce_finit_type == REDUCE_FINIT_SCALE) {
            ff = outtype == CV_32F && worktype == CV_64F ? reduce_finit_scale_64f32f :
                outtype == CV_16F && worktype == CV_32F ? reduce_finit_scale_32f16f :
                outtype == CV_16BF && worktype == CV_32F ? reduce_finit_scale_32f16f : nullptr;
        } else if (reduce_finit_type == REDUCE_FINIT_SQRT) {
            ff = outtype == CV_32F && worktype == CV_64F ? reduce_finit_sqrt_64f32f :
                outtype == CV_16F && worktype == CV_64F ? reduce_finit_sqrt_32f16f :
                outtype == CV_16BF && worktype == CV_64F ? reduce_finit_sqrt_32f16f : nullptr;
        }
        if (!rf || !ff)
            CV_Error(Error::StsNotImplemented, "");
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

        int inptype = inp.type(), outtype = inferType(inptype), worktype = outtype;
        TensorSize inpsize0 = inp.size(), outsize0, outsize0_kd, inpsize, outsize;
        size_t inpstep[TensorSize::MAX_DIMS];
        const Tensor& axes_ = inputs[1];
        bool reduce_mask[TensorSize::MAX_DIMS];
        int axes[TensorSize::MAX_DIMS];
        int naxes = inferShapes_(inpsize0, axes_, axes, outsize0, outsize0_kd, reduce_mask);
        outputs.resize(1);
        Tensor& out = outputs[0];
        out.fitSameDevice(inp, outsize0, outtype);
        CV_Assert(out.isContinuous());

        /*
            'compress' dimensionality of the input and output arrays.
            Reduce op can be written as:
            out[i0, i1, ..., i(n-1)] = reduce_op_{j0,j1,...} inp[i0,...ik1,j0,..,jk2,i(k1+1),...,ik2,...]
            when keepdims = false.
            when keepdims = true then the left part should be replaced with
            out[i0,0,0...,0,i1,0,...,0,i2,...,i(n-1)]
            that is, 0-values are inserted to equalize 'out' and 'inp' dimensionality

            the N-D index in inp[...] expression is the interleaved list of output (fixed) indices and
            varying indices along which reduction is performed.

            The compression loop below replaces several subsequent i*,...,i* indices without any 'j' in the middle
            with a single index, and also replaces several subsequent j*,...,j* indices without any 'i' in the middle
            with a single index, so the formula becomes

            out[I0,I1,...] = reduce_op_{J0,J1,...} inp[I0,J0,I1,J1,....]
            where I* and J* are compressed indices. inp[...] may end with some J* or some I*.
            Since both input and output are contiguous, the scheme lets us to
            significantly speed up indices calculations and decrease the number of nested loops.
            E.g. with a single reduction axis we get at most 3D loop:
            out[I0,I1] = reduce_op_{j} inp[I0,j,I1]
            or even 2D loop:
            out[I] = reduce_op_{j} inp[I,j] or
            out[I] = reduce_op_{j} inp[j,I].
        */

        inpsize.ndims = 0;
        int64_t compressed_dim_size = 1;
        bool reduce_mode = reduce_mask[0];
        for (int j = 0; j < inpsize0.ndims; j++) {
            bool mj = reduce_mask[j];
            if (mj == reduce_mode)
                compressed_dim_size *= inpsize0.size[j];
            else {
                inpsize.size[inpsize.ndims] = compressed_dim_size;
                outsize.size[inpsize.ndims++] = reduce_mode ? 1 : compressed_dim_size;
                reduce_mode = mj;
                compressed_dim_size = inpsize0.size[j];
            }
        }
        inpsize.size[inpsize.ndims] = compressed_dim_size;
        outsize.size[inpsize.ndims++] = reduce_mode ? 1 : compressed_dim_size;
        for (; inpsize.ndims > 1; inpsize.ndims--) {
            if (inpsize.size[inpsize.ndims-1] > 1)
                break;
        }
        int ndims = inpsize.ndims;
        outsize.ndims = ndims;

        inpstep[ndims-1] = 1;
        for (int j = ndims-2; j >= 0; j--) {
            inpstep[j] = inpstep[j+1]*inpsize.size[j+1];
        }

        /*printf("inpsize after compression: ");
        for (int j = 0; j < inpsize.ndims; j++)
            printf("%zu ", (size_t)inpsize.size[j]);
        printf("\noutsize after compression: ");
        for (int j = 0; j < outsize.ndims; j++)
            printf("%zu ", (size_t)outsize.size[j]);
        printf("\n");*/

        int64_t ninp = (int64_t)inpsize.total();
        int64_t nout = (int64_t)outsize.total();
        if (nout == 0)
            return;
        int64_t reduce_size = ninp/nout;
        double param = 0.;
        reduce_func_t reduce_func = nullptr;
        reduce_finit_func_t reduce_finit_func = nullptr;

        getReduceFunc(inptype, outtype, opcode, worktype, reduce_func, reduce_finit_func);

        if (opcode == REDUCE_MEAN)
            param = 1./std::max((double)reduce_size, 1.);

        const int64_t BLOCK_SIZE = 1 << 17;
        constexpr int64_t MINI_BLOCK_SIZE = 32;
        // 'reduce' is the family of functions which complexity is proportional to the number of input elements,
        // so we compute the number of blocks (that could be processed in parallel) based on this number
        int64_t nblocks = ((int64_t)ninp + BLOCK_SIZE-1)/BLOCK_SIZE;

        nblocks = std::min(nblocks, nout);
        const char* inptr0 = (const char*)inputs[0].data();
        char* outptr0 = (char*)outputs[0].data();
        size_t esz = CV_ELEM_SIZE(inptype);

        parallel_for_(Range(0, (int)nblocks), [&](const Range& r) {
            int64_t start = r.start*nout/nblocks;
            int64_t end = r.end*nout/nblocks;
            int64_t blocksize = MINI_BLOCK_SIZE;
            int64_t accbuf[MINI_BLOCK_SIZE];
            int rdim = -(outsize.size[ndims-1] == 1);
            int64_t ncols0 = inpsize.size[ndims-1];
            int64_t nrows0 = ndims >= 2 ? inpsize.size[ndims-2] : 1;
            int64_t planesize = nrows0*ncols0;
            int64_t dk = rdim == 0 ? nrows0 : ncols0;
            int64_t nout0 = planesize / dk;

            for (int64_t outofs = start; outofs < end; outofs += blocksize) {
                int64_t idx = outofs, inpofs = 0;
                int64_t x = idx % nout0;
                blocksize = std::min(MINI_BLOCK_SIZE, nout0 - x);
                blocksize = std::min(blocksize, end - outofs);
                int64_t nrows = nrows0, ncols = ncols0;
                if (rdim == 0)
                    ncols = blocksize;
                else
                    nrows = blocksize;

                for (int j = ndims-1; j >= 0; j--) {
                    int64_t szj = outsize.size[j];
                    int64_t pidx = idx / szj;
                    inpofs += inpstep[j]*(idx - pidx*szj);
                    idx = pidx;
                }

                for (int64_t k = 0; k < reduce_size; k += dk, inpofs += planesize) {
                    reduce_func(inptr0 + inpofs*esz, ncols0, accbuf,
                                nrows, ncols, rdim, k == 0);
                }
                reduce_finit_func(accbuf, outptr0 + outofs*esz, blocksize, &param);
            }
        });
    }

    bool haveFP16;
};

Op ReduceOp::create(ReduceOpcode opcode, bool keepdims, bool noOpWithEmptyAxes)
{
    return std::make_shared<ReduceOpImpl>(opcode, keepdims, noOpWithEmptyAxes);
}

}}
