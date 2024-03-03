// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"
#include <math.h>

namespace cv { namespace dnn {

#undef CV_IMPLEMENT_MATH_OP
#define CV_IMPLEMENT_MATH_OP(name, op) \
static void elemwise_##name##_f32(size_t ninputs, const void** inptr_, const size_t* steps, \
                                  void* outptr_, size_t len, const float*) \
{ \
    CV_Assert(ninputs == 1 && steps[0] == 1); \
    const float* inptr = (const float*)inptr_[0]; \
    float* outptr = (float*)outptr_; \
    for (size_t j = 0; j < len; j++) { \
        float x = inptr[j]; \
        outptr[j] = (float)op(x); \
    } \
} \
\
static void elemwise_##name##_f16(size_t ninputs, const void** inptr_, const size_t* steps, \
                                void* outptr_, size_t len, const float*) \
{ \
    CV_Assert(ninputs == 1 && steps[0] == 1); \
    const cv::float16_t* inptr = (const cv::float16_t*)inptr_[0]; \
    cv::float16_t* outptr = (cv::float16_t*)outptr_; \
    for (size_t j = 0; j < len; j++) { \
        float x = (float)inptr[j]; \
        outptr[j] = cv::float16_t(op(x)); \
    } \
}

#define CV_RELU(x) std::max((x), 0.f)
#define CV_SIGN(x) (((x)>0.f)-((x)<0.f))

CV_IMPLEMENT_MATH_OP(abs, fabsf)
CV_IMPLEMENT_MATH_OP(acos, acosf)
CV_IMPLEMENT_MATH_OP(acosh, acoshf)
CV_IMPLEMENT_MATH_OP(asin, asinf)
CV_IMPLEMENT_MATH_OP(asinh, asinhf)
CV_IMPLEMENT_MATH_OP(atan, atanf)
CV_IMPLEMENT_MATH_OP(atanh, atanhf)
CV_IMPLEMENT_MATH_OP(ceil, ceilf)
CV_IMPLEMENT_MATH_OP(cos, cosf)
CV_IMPLEMENT_MATH_OP(cosh, coshf)
CV_IMPLEMENT_MATH_OP(erf, erff)
CV_IMPLEMENT_MATH_OP(exp, expf)
CV_IMPLEMENT_MATH_OP(floor, floorf)
CV_IMPLEMENT_MATH_OP(log, logf)
CV_IMPLEMENT_MATH_OP(relu, CV_RELU)
CV_IMPLEMENT_MATH_OP(round, roundf)
CV_IMPLEMENT_MATH_OP(sign, CV_SIGN)
CV_IMPLEMENT_MATH_OP(sin, sinf)
CV_IMPLEMENT_MATH_OP(sinh, sinhf)
CV_IMPLEMENT_MATH_OP(sqrt, sqrtf)
CV_IMPLEMENT_MATH_OP(tan, tanf)
CV_IMPLEMENT_MATH_OP(tanh,  tanhf)

static void elemwise_clip_f32(size_t ninputs, const void** inptr_, const size_t* steps,
                              void* outptr_, size_t len, const float*)
{
    float minv = -FLT_MAX, maxv = FLT_MAX;
    CV_Assert(steps[0] == 1);
    if (ninputs > 1)
        minv = *((const float*)inptr_[1]);
    if (ninputs > 2)
        maxv = *((const float*)inptr_[2]);
    const float* inptr = (const float*)inptr_[0];
    float* outptr = (float*)outptr_;
    for (size_t j = 0; j < len; j++) {
        float x = inptr[j];
        outptr[j] = (float)std::min(std::max(x, minv), maxv);
    }
}

static void elemwise_clip_f16(size_t ninputs, const void** inptr_, const size_t* steps,
                              void* outptr_, size_t len, const float*)
{
    float minv = -65504.f, maxv = 65504.f;
    CV_Assert(steps[0] == 1);
    if (ninputs > 1)
        minv = *((const float*)inptr_[1]);
    if (ninputs > 2)
        maxv = *((const float*)inptr_[2]);
    const cv::float16_t* inptr = (const cv::float16_t*)inptr_[0];
    cv::float16_t* outptr = (cv::float16_t*)outptr_;
    for (size_t j = 0; j < len; j++) {
        float x = (float)inptr[j];
        outptr[j] = cv::float16_t(std::min(std::max(x, minv), maxv));
    }
}

static void elemwise_leaky_relu_f32(size_t ninputs, const void** inptr_, const size_t* steps,
                                    void* outptr_, size_t len, const float* params)
{
    float alpha = params[0];
    CV_Assert(ninputs == 1 && steps[0] == 1);
    const float* inptr = (const float*)inptr_[0];
    float* outptr = (float*)outptr_;
    for (size_t j = 0; j < len; j++) {
        float x = inptr[j];
        outptr[j] = (float)(x*(x < 0 ? alpha : 1.f));
    }
}

static void elemwise_leaky_relu_f16(size_t ninputs, const void** inptr_, const size_t* steps,
                                    void* outptr_, size_t len, const float* params)
{
    float alpha = params[0];
    CV_Assert(ninputs == 1 && steps[0] == 1);
    const cv::float16_t* inptr = (const cv::float16_t*)inptr_[0];
    cv::float16_t* outptr = (cv::float16_t*)outptr_;
    for (size_t j = 0; j < len; j++) {
        float x = (float)inptr[j];
        outptr[j] = cv::float16_t(x*(x < 0 ? alpha : 1.f));
    }
}

#undef CV_IMPLEMENT_BINARY_OP
#define CV_IMPLEMENT_BINARY_OP(name, suffix, T1, T2, WT, op) \
static void elemwise_##name##_##suffix(size_t ninputs, const void** inptr_, const size_t* steps, \
                                       void* outptr_, size_t len, const float*) \
{ \
    CV_Assert(ninputs == 2); \
    size_t step0 = steps[0], step1 = steps[1]; \
    const T1* inptr0 = (const T1*)inptr_[0]; \
    const T1* inptr1 = (const T1*)inptr_[1]; \
    T2* outptr = (T2*)outptr_; \
    if (step0 == 1 && step1 == 1) { \
        for (size_t j = 0; j < len; j++) { \
            WT x0 = (WT)inptr0[j], x1 = (WT)inptr1[j]; \
            outptr[j] = saturate_cast<T2>(op(x0, x1)); \
        } \
    } else if (step0 == 1 && step1 == 0) { \
        WT x1 = (WT)*inptr1; \
        for (size_t j = 0; j < len; j++) { \
            WT x0 = (WT)inptr0[j]; \
            outptr[j] = saturate_cast<T2>(op(x0, x1)); \
        } \
    } else if (step0 == 0 && step1 == 1) { \
        WT x0 = (WT)*inptr0; \
        for (size_t j = 0; j < len; j++) { \
            WT x1 = (WT)inptr1[j]; \
            outptr[j] = saturate_cast<T2>(op(x0, x1)); \
        } \
    } else { \
        for (size_t j = 0; j < len; j++, inptr0 += step0, inptr1 += step1) { \
            WT x0 = (WT)*inptr0, x1 = (WT)*inptr1; \
            outptr[j] = saturate_cast<T2>(op(x0, x1)); \
        } \
    } \
}

#define CV_IMPLEMENT_ARITHM_OP_ALLTYPES(name, iop, fop) \
    CV_IMPLEMENT_BINARY_OP(name, 8u, uint8_t, uint8_t, uint8_t, iop) \
    CV_IMPLEMENT_BINARY_OP(name, 8s, int8_t, int8_t, int8_t, iop) \
    CV_IMPLEMENT_BINARY_OP(name, 16u, uint16_t, uint16_t, uint16_t, iop) \
    CV_IMPLEMENT_BINARY_OP(name, 16s, int16_t, int16_t, int16_t, iop) \
    CV_IMPLEMENT_BINARY_OP(name, 32u, uint32_t, uint32_t, int64_t, iop) \
    CV_IMPLEMENT_BINARY_OP(name, 32s, int32_t, int32_t, int32_t, iop) \
    CV_IMPLEMENT_BINARY_OP(name, 64u, uint64_t, uint64_t, uint64_t, iop) \
    CV_IMPLEMENT_BINARY_OP(name, 64s, int64_t, int64_t, int64_t, iop) \
    CV_IMPLEMENT_BINARY_OP(name, 32f, float, float, float, fop) \
    CV_IMPLEMENT_BINARY_OP(name, 16f, cv::float16_t, cv::float16_t, float, fop)

#define CV_IMPLEMENT_LOGIC_OP_ALLTYPES(name, op) \
    CV_IMPLEMENT_BINARY_OP(name, 8u, uint8_t, uint8_t, uint8_t, op) \
    CV_IMPLEMENT_BINARY_OP(name, 16u, uint16_t, uint16_t, uint16_t, op) \
    CV_IMPLEMENT_BINARY_OP(name, 32u, uint32_t, uint32_t, uint32_t, op) \
    CV_IMPLEMENT_BINARY_OP(name, 64u, uint64_t, uint64_t, uint64_t, op)

#define CV_IMPLEMENT_CMP_ALLTYPES(name, op) \
    CV_IMPLEMENT_BINARY_OP(name, 8u, uint8_t, bool, uint8_t, op) \
    CV_IMPLEMENT_BINARY_OP(name, 8s, int8_t, bool, int8_t, op) \
    CV_IMPLEMENT_BINARY_OP(name, 16u, uint16_t, bool, uint16_t, op) \
    CV_IMPLEMENT_BINARY_OP(name, 16s, int16_t, bool, int16_t, op) \
    CV_IMPLEMENT_BINARY_OP(name, 32u, uint32_t, bool, uint32_t, op) \
    CV_IMPLEMENT_BINARY_OP(name, 32s, int32_t, bool, int32_t, op) \
    CV_IMPLEMENT_BINARY_OP(name, 64u, uint64_t, bool, uint64_t, op) \
    CV_IMPLEMENT_BINARY_OP(name, 64s, int64_t, bool, int64_t, op) \
    CV_IMPLEMENT_BINARY_OP(name, 32f, float, bool, float, op) \
    CV_IMPLEMENT_BINARY_OP(name, 16f, cv::float16_t, bool, float, op)

#define CV_IMPLEMENT_CMP_STUB_OP(name0, name1, suffix, T) \
static void elemwise_##name0##_##suffix(size_t ninputs, const void** inptr_, const size_t* steps, \
                                        void* outptr_, size_t len, const float* params) \
{ \
    CV_Assert(ninputs == 2); \
    const void* inptr_swp[] = {inptr_[1], inptr_[0]}; \
    const size_t steps_swp[] = {steps[1], steps[0]}; \
    elemwise_##name1##_##suffix(ninputs, inptr_swp, steps_swp, outptr_, len, params); \
}

#define CV_IMPLEMENT_CMP_STUB_ALLTYPES(name0, name1) \
    CV_IMPLEMENT_CMP_STUB_OP(name0, name1, 8u, uint8_t) \
    CV_IMPLEMENT_CMP_STUB_OP(name0, name1, 8s, int8_t) \
    CV_IMPLEMENT_CMP_STUB_OP(name0, name1, 16u, uint16_t) \
    CV_IMPLEMENT_CMP_STUB_OP(name0, name1, 16s, int16_t) \
    CV_IMPLEMENT_CMP_STUB_OP(name0, name1, 32u, uint32_t) \
    CV_IMPLEMENT_CMP_STUB_OP(name0, name1, 32s, int32_t) \
    CV_IMPLEMENT_CMP_STUB_OP(name0, name1, 64u, uint64_t) \
    CV_IMPLEMENT_CMP_STUB_OP(name0, name1, 64s, int64_t) \
    CV_IMPLEMENT_CMP_STUB_OP(name0, name1, 32f, float) \
    CV_IMPLEMENT_CMP_STUB_OP(name0, name1, 16f, cv::float16_t)

#undef CV_ADD
#define CV_ADD(x, y) ((x) + (y))
#undef CV_SUB
#define CV_SUB(x, y) ((x) - (y))
#undef CV_MUL
#define CV_MUL(x, y) ((x) * (y))
#undef CV_DIV
#define CV_DIV(x, y) ((x) / (y))
#undef CV_MOD
#define CV_MOD(x, y) ((x) % (y))
#undef CV_AND
#define CV_AND(x, y) ((x) & (y))
#undef CV_OR
#define CV_OR(x, y) ((x) | (y))
#undef CV_XOR
#define CV_XOR(x, y) ((x) ^ (y))

CV_IMPLEMENT_ARITHM_OP_ALLTYPES(add, CV_ADD, CV_ADD)
CV_IMPLEMENT_ARITHM_OP_ALLTYPES(sub, CV_SUB, CV_SUB)
CV_IMPLEMENT_ARITHM_OP_ALLTYPES(mul, CV_MUL, CV_MUL)
CV_IMPLEMENT_ARITHM_OP_ALLTYPES(div, CV_DIV, CV_DIV)
CV_IMPLEMENT_ARITHM_OP_ALLTYPES(mod, CV_MOD, fmodf)
CV_IMPLEMENT_ARITHM_OP_ALLTYPES(max, std::max, std::max)
CV_IMPLEMENT_ARITHM_OP_ALLTYPES(min, std::min, std::min)

CV_IMPLEMENT_LOGIC_OP_ALLTYPES(and, CV_AND)
CV_IMPLEMENT_LOGIC_OP_ALLTYPES(or, CV_OR)
CV_IMPLEMENT_LOGIC_OP_ALLTYPES(xor, CV_XOR)

#undef CV_EQ
#define CV_EQ(x, y) ((x) == (y))
#undef CV_NE
#define CV_NE(x, y) ((x) != (y))
#undef CV_LE
#define CV_LE(x, y) ((x) <= (y))
#undef CV_LT
#define CV_LT(x, y) ((x) < (y))

CV_IMPLEMENT_CMP_ALLTYPES(eq, CV_EQ)
CV_IMPLEMENT_CMP_ALLTYPES(ne, CV_NE)
CV_IMPLEMENT_CMP_ALLTYPES(le, CV_LE)
CV_IMPLEMENT_CMP_ALLTYPES(lt, CV_LT)

CV_IMPLEMENT_CMP_STUB_ALLTYPES(ge, le)
CV_IMPLEMENT_CMP_STUB_ALLTYPES(gt, lt)

std::string_view elemwiseOpcode2str(ElemwiseOpcode opcode_)
{
    constexpr int nopcodes = (int)ELWISE_OPCODE_MAX;
    static const char* names[nopcodes];
    static volatile bool initialized = false;

    if (!initialized) {
        names[ELWISE_NONE] = "<none>";
        names[ELWISE_ADD] = "Add";
        names[ELWISE_AND] = "And";
        names[ELWISE_DIV] = "Div";
        names[ELWISE_EQUAL] = "Equal";
        names[ELWISE_GREATER] = "Greater";
        names[ELWISE_GREATER_EQUAL] = "GreaterEqual";
        names[ELWISE_LESS] = "Less";
        names[ELWISE_LESS_EQUAL] = "LessEqual";
        names[ELWISE_MAX] = "Max";
        names[ELWISE_MEAN] = "Mean";
        names[ELWISE_MIN] = "Min";
        names[ELWISE_MOD] = "Mod";
        names[ELWISE_MUL] = "Mul";
        names[ELWISE_POW] = "Pow";
        names[ELWISE_OR] = "Or";
        names[ELWISE_SUB] = "Sub";
        names[ELWISE_SUM] = "Sum";
        names[ELWISE_XOR] = "Xor";

        names[ELWISE_ABS] = "Abs";
        names[ELWISE_ACOS] = "Acos";
        names[ELWISE_ACOSH] = "Acosh";
        names[ELWISE_ASIN] = "Asin";
        names[ELWISE_ASINH] = "Asinh";
        names[ELWISE_ATAN] = "Atan";
        names[ELWISE_ATANH] = "Atanh";
        names[ELWISE_CEIL] = "Ceil";
        names[ELWISE_CLIP] = "Clip";
        names[ELWISE_COS] = "Cos";
        names[ELWISE_COSH] = "Cosh";
        names[ELWISE_ERF] = "Erf";
        names[ELWISE_EXP] = "Exp";
        names[ELWISE_FLOOR] = "Floor";
        names[ELWISE_ISINF] = "IsInf";
        names[ELWISE_ISNAN] = "IsNan";
        names[ELWISE_LOG] = "Log";
        names[ELWISE_LRELU] = "LeakyRelu";
        names[ELWISE_MISH] = "Mish";
        names[ELWISE_NEG] = "Neg";
        names[ELWISE_NOT] = "Not";
        names[ELWISE_RELU] = "Relu";
        names[ELWISE_ROUND] = "Round";
        names[ELWISE_SIGMOID] = "Sigmoid";
        names[ELWISE_SIGN] = "Sign";
        names[ELWISE_SIN] = "Sin";
        names[ELWISE_SINH] = "Sinh";
        names[ELWISE_SOFTPLUS] = "Softplus";
        names[ELWISE_SOFTSIGN] = "Softsign";
        names[ELWISE_SQRT] = "Sqrt";
        names[ELWISE_TAN] = "Tan";
        names[ELWISE_TANH] = "Tanh";
        initialized = true;
    }
    int opcode = (int)opcode_;
    CV_Assert(0 <= opcode && opcode < (int)ELWISE_OPCODE_MAX);
    return names[opcode];
}

ElemwiseOp::~ElemwiseOp() {}

class ElemwiseOpImpl : public ElemwiseOp
{
public:
    ElemwiseOpImpl(ElemwiseOpcode opcode_, const float* params_, size_t nparams_)
    {
        opcode = opcode_;
        size_t i = 0;
        if (params_) {
            CV_Assert(nparams_ <= MAX_PARAMS);
            for (; i < nparams_; i++)
                params[i] = params_[i];
        }
        for (; i < MAX_PARAMS; i++)
            params[i] = 0.f;

        haveFP16 = checkHardwareSupport(CV_CPU_FP16);
    }
    virtual std::string_view name() const CV_OVERRIDE { return elemwiseOpcode2str(opcode); }
    virtual Op clone() const CV_OVERRIDE
    {
        return std::make_shared<ElemwiseOpImpl>(opcode, params, MAX_PARAMS);
    }

    virtual int minNumInputs() const CV_OVERRIDE
    {
        if (opcode == ELWISE_ADD ||
            opcode == ELWISE_AND ||
            opcode == ELWISE_DIV ||
            opcode == ELWISE_EQUAL ||
            opcode == ELWISE_GREATER ||
            opcode == ELWISE_GREATER_EQUAL ||
            opcode == ELWISE_LESS ||
            opcode == ELWISE_LESS_EQUAL ||
            opcode == ELWISE_MAX ||
            opcode == ELWISE_MEAN ||
            opcode == ELWISE_MIN ||
            opcode == ELWISE_MOD ||
            opcode == ELWISE_MUL ||
            opcode == ELWISE_POW ||
            opcode == ELWISE_OR ||
            opcode == ELWISE_SUB ||
            opcode == ELWISE_SUM ||
            opcode == ELWISE_XOR)
            return 2;
        return 1;
    }
    virtual int maxNumInputs() const CV_OVERRIDE
    {
        if (opcode == ELWISE_CLIP)
            return 3;
        if (opcode == ELWISE_MAX || opcode == ELWISE_MEAN ||
            opcode == ELWISE_MIN || opcode == ELWISE_SUM)
            return INT_MAX;
        return minNumInputs();
    }
    virtual int minNumOutputs() const CV_OVERRIDE { return 1; }
    virtual int maxNumOutputs() const CV_OVERRIDE { return 1; }

    int inferType(int inptype0) const
    {
        CV_Assert(supportType(0, inptype0));
        if (opcode == ELWISE_EQUAL ||
            opcode == ELWISE_GREATER ||
            opcode == ELWISE_GREATER_EQUAL ||
            opcode == ELWISE_LESS ||
            opcode == ELWISE_LESS_EQUAL ||
            opcode == ELWISE_ISINF ||
            opcode == ELWISE_ISNAN)
            return CV_Bool;

        return inptype0;
    }

    virtual bool supportType(int, int depth) const CV_OVERRIDE
    {
        if (depth == CV_32F || (depth == CV_16F && haveFP16))
            return true;
        if (depth == CV_Bool &&
            (opcode == ELWISE_AND ||
             opcode == ELWISE_OR ||
             opcode == ELWISE_XOR ||
             opcode == ELWISE_NOT))
            return true;
        if (isIntType(depth) &&
            (opcode == ELWISE_ADD ||
            opcode == ELWISE_AND ||
            opcode == ELWISE_DIV ||
            opcode == ELWISE_EQUAL ||
            opcode == ELWISE_GREATER ||
            opcode == ELWISE_GREATER_EQUAL ||
            opcode == ELWISE_LESS ||
            opcode == ELWISE_LESS_EQUAL ||
            opcode == ELWISE_MAX ||
            opcode == ELWISE_MEAN ||
            opcode == ELWISE_MIN ||
            opcode == ELWISE_MOD ||
            opcode == ELWISE_MUL ||
            opcode == ELWISE_POW ||
            opcode == ELWISE_OR ||
            opcode == ELWISE_SUB ||
            opcode == ELWISE_SUM ||
            opcode == ELWISE_XOR ||

            opcode == ELWISE_CLIP ||
            opcode == ELWISE_NOT ||
            opcode == ELWISE_RELU))
            return true;

        return false;
    }

    virtual bool alwaysSupportInplace() const CV_OVERRIDE
    {
        int inptype = CV_32F;
        return opcode == ELWISE_CLIP || (maxNumInputs() == 1 && inferType(inptype) == CV_32F);
    }

    virtual int64_t getFLOPS(const std::vector<SizeType> &inputs,
                           const std::vector<SizeType> &outputs) const CV_OVERRIDE
    {
        CV_Assert(outputs.size() == 1);
        // probably, there should be a coefficient in the case of complex math functions,
        // like ~10-20 for exp, log, tanh, sigmoid, ...
        return (int64_t)outputs[0].size.total();
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
        outst[0].size = inpst[0].size;
        outst[0].type = inferType(inpst[0].type);
        for (int i = 1; i < ninputs; i++) {
            // all element-wise operations we support so far
            // require all input parameters to have the same type
            CV_Assert(inpst[i].type == inpst[0].type);
            outst[0].size = outst[0].size.expand(inpst[i].size);
        }

        tempbufs.assign(1, (size_t)0);
    }

    virtual forward_t getForwardSlice(int type) const CV_OVERRIDE
    {

    }

    virtual void forward(Net2& net, Graph& graph,
                        const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs,
                        std::vector<Buffer>& tempbufs) CV_OVERRIDE
    {
        CV_Error(Error::StsNotImplemented, "");
    }

    bool haveFP16;
};

Op ElemwiseOp::create(ElemwiseOpcode opcode,
                      const float* params, size_t nparams)
{
    return std::make_shared<ElemwiseOpImpl>(opcode, params, nparams);
}

}}
