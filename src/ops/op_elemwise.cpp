// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"
#include <math.h>

namespace cv { namespace dnn {

/*
    Prepare the strides for the efficient max_ndims-dimensional operation.
    The algorithm includes the following steps:

    1. Make both input array and the output array max_ndims-dimensional.
       If necessary, prepend the shapes with 1's and compute the corresponding strides.
       This makes the actual operation much more straight-forward,
       we just have to deal with fixed dimensionality using fixed number of nested loops.
    2. For some i one of the inputs may have i-th dimension == 1,
       whereas the other input may have the same dimension > 1.
       We need to handle it by zero'ing the corresponding i-th step.

       E.g. size1[2] == 1, size2[2] == 100 (and so size[2] == 100).

       When we will iterate through this dimension within a nested loop and access elements

       for (int i0 = 0; i0 < shape[0]; i0++)
          for (int i1 = 0; i1 < shape[1]; i++) {
             for (int i2 = 0; i2 < shape[2]; i2++) {
                ...
                input1.ptr<float>[i0*step1[0] + i1*step1[1] + i2*step1[2] + ...]
             }

       we need to take into account that shape1[2] == 1 and set step1[2]=0.
    3. Often the inputs are contiguous (the output is assumed to be contiguous),
       so we can try to flatten/reshape inputs in order to increase the length of inner loops and
       correspondingly shorten the outer loops so that the loop overhead is reduced.
       We do flattening within in a loop with descending j, starting from j=max_ndims-2:
       3a. we check that for some stepi[j] = stepi[j+1]*sizei[j+1] for all i: i=0,1,2 (i=0 means the output tensor)
       3b. we also check that for each tensor we stay in scalar mode or stay in non-scalar mode
       3c. we also check if sizei[j] == 1 for all i.
       3d. if yes for (3a && (3b || 3c)), we do shapei[j+1] *= shapei[j] and eliminate j-th dimension.
       3e. otherwise, we leave everything as is, decrease j and proceed.
       3f. in the end of the loop we append the proper number of 1's
           to the final shape to keep it max_ndims-dimensional.
*/
int prepareForBroadcasting(
    int ntensors, const TensorSize* sizes0,
    TensorSize* sizes, size_t** steps)
{
    int max_ndims = 1;
    int i, j, k;
    for (k = 0; k < ntensors; k++)
        max_ndims = std::max(max_ndims, sizes0[k].ndims);

    // step 1.
    // * make all inputs and the output max_ndims-dimensional.
    // * compute proper step's
    for (i = max_ndims-1; i >= 0; i-- ) {
        for (k = 0; k < ntensors; k++) {
            j = sizes0[k].ndims - (max_ndims - i);
            int64_t sz_i = j >= 0 ? sizes0[k].size[j] : 1;
            size_t st_i = i == max_ndims-1 ? 1 : steps[k][i+1]*sizes[k].size[i+1];
            sizes[k].size[i] = sz_i;
            steps[k][i] = st_i;
            if (sizes[k].size[i] == 0)
                return -1;
        }
    }

    // step 3. Let's do the flattening first,
    // since we'd need proper values of steps to check continuity.
    // this loop is probably the most tricky part
    // in the whole implementation of broadcasting.
    j = max_ndims-1;
    for (i = j - 1; i >= 0; i--) {
        bool all_contiguous = true, all_scalars = true, all_consistent = true;
        for(k = 0; k < ntensors; k++) {
            size_t st = steps[k][j]*sizes[k].size[j];
            bool prev_scalar = sizes[k].size[j] == 1;
            bool scalar = sizes[k].size[i] == 1;
            all_contiguous = all_contiguous && (st == steps[k][i]);
            all_scalars = all_scalars && scalar;
            all_consistent = all_consistent && (scalar == prev_scalar);
        }
        if (all_contiguous && (all_consistent || all_scalars)) {
            for(k = 0; k < ntensors; k++)
                sizes[k].size[j] *= sizes[k].size[i];
        } else {
            j--;
            if (i < j) {
                for(k = 0; k < ntensors; k++) {
                    sizes[k].size[j] = sizes[k].size[i];
                    steps[k][j] = steps[k][i];
                }
            }
        }
    }

    // step 2. Set some step's to 0's.
    for (i = max_ndims-1; i >= j; i--) {
        for (k = 0; k < ntensors; k++)
            steps[k][i] = sizes[k].size[i] == 1 ? 0 : steps[k][i];
    }

    for (; i >= 0; i--) {
        for (k = 0; k < ntensors; k++) {
            steps[k][i] = 0;
            sizes[k].size[i] = 1;
        }
    }

    for (k = 0; k < ntensors; k++) {
        sizes[k].ndims = max_ndims;
        sizes[k].layout = LAYOUT_UNKNOWN;
    }

    return max_ndims;
}

#undef CV_IMPLEMENT_UNARY_OP
#define CV_IMPLEMENT_UNARY_OP(name, suffix, T1, T2, WT, op) \
static void elemwise_##name##_##suffix(size_t ninputs, const void** inptr_, const size_t* steps, \
                                       void* outptr_, size_t len, const float*) \
{ \
    CV_Assert(ninputs == 1 && steps[0] == 1); \
    const T1* inptr = (const T1*)inptr_[0]; \
    T2* outptr = (T2*)outptr_; \
    for (size_t j = 0; j < len; j++) { \
        WT x = (WT)inptr[j]; \
        outptr[j] = T2(op(x)); \
    } \
}

#undef CV_IMPLEMENT_MATH_OP
#define CV_IMPLEMENT_MATH_OP(name, op) \
    CV_IMPLEMENT_UNARY_OP(name, 32f, float, float, float, op) \
    CV_IMPLEMENT_UNARY_OP(name, 16f, cv::float16_t, cv::float16_t, float, op)

#define CV_RELU(x) std::max((x), 0.f)
#define CV_SIGN(x) (((x)>0.f)-((x)<0.f))
#define CV_SIGMOID(x) (1.f/(1 + expf(-(x))))
#define CV_NOT(x) (~(x))
#define CV_NEG(x) (-(x))

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
CV_IMPLEMENT_MATH_OP(sigmoid, CV_SIGMOID)
CV_IMPLEMENT_MATH_OP(sign, CV_SIGN)
CV_IMPLEMENT_MATH_OP(sin, sinf)
CV_IMPLEMENT_MATH_OP(sinh, sinhf)
CV_IMPLEMENT_MATH_OP(sqrt, sqrtf)
CV_IMPLEMENT_MATH_OP(tan, tanf)
CV_IMPLEMENT_MATH_OP(tanh,  tanhf)

CV_IMPLEMENT_UNARY_OP(not, 8u, uint8_t, uint8_t, int, CV_NOT)
CV_IMPLEMENT_UNARY_OP(not, 16u, uint16_t, uint16_t, int, CV_NOT)
CV_IMPLEMENT_UNARY_OP(not, 32u, uint32_t, uint32_t, uint32_t, CV_NOT)
CV_IMPLEMENT_UNARY_OP(not, 64u, uint64_t, uint64_t, uint64_t, CV_NOT)

CV_IMPLEMENT_UNARY_OP(neg, 8s, int8_t, int8_t, int8_t, CV_NEG)
CV_IMPLEMENT_UNARY_OP(neg, 16s, int16_t, int16_t, int16_t, CV_NEG)
CV_IMPLEMENT_UNARY_OP(neg, 32s, int32_t, int32_t, int32_t, CV_NEG)
CV_IMPLEMENT_UNARY_OP(neg, 64s, int64_t, int64_t, int64_t, CV_NEG)
CV_IMPLEMENT_UNARY_OP(neg, 32f, float, float, float, CV_NEG)
CV_IMPLEMENT_UNARY_OP(neg, 16f, cv::float16_t, cv::float16_t, float, CV_NEG)

static void elemwise_clip_32f(size_t ninputs, const void** inptr_, const size_t* steps,
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

static void elemwise_clip_16f(size_t ninputs, const void** inptr_, const size_t* steps,
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

static void elemwise_leaky_relu_32f(size_t ninputs, const void** inptr_, const size_t* steps,
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

static void elemwise_leaky_relu_16f(size_t ninputs, const void** inptr_, const size_t* steps,
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
#define CV_IMPLEMENT_BINARY_OP(name, suffix, T1, T2, WT, op, init) \
static void elemwise_##name##_##suffix(size_t ninputs, const void** inptr_, const size_t* steps, \
                                       void* outptr_, size_t len, const float* params) \
{ \
    init; \
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

#define CV_IMPLEMENT_ARITHM_OP_ALLTYPES(name, iop, fop, init) \
    CV_IMPLEMENT_BINARY_OP(name, 8u, uint8_t, uint8_t, uint8_t, iop, init) \
    CV_IMPLEMENT_BINARY_OP(name, 8s, int8_t, int8_t, int8_t, iop, init) \
    CV_IMPLEMENT_BINARY_OP(name, 16u, uint16_t, uint16_t, uint16_t, iop, init) \
    CV_IMPLEMENT_BINARY_OP(name, 16s, int16_t, int16_t, int16_t, iop, init) \
    CV_IMPLEMENT_BINARY_OP(name, 32u, uint32_t, uint32_t, int64_t, iop, init) \
    CV_IMPLEMENT_BINARY_OP(name, 32s, int32_t, int32_t, int32_t, iop, init) \
    CV_IMPLEMENT_BINARY_OP(name, 64u, uint64_t, uint64_t, uint64_t, iop, init) \
    CV_IMPLEMENT_BINARY_OP(name, 64s, int64_t, int64_t, int64_t, iop, init) \
    CV_IMPLEMENT_BINARY_OP(name, 32f, float, float, float, fop, init) \
    CV_IMPLEMENT_BINARY_OP(name, 16f, cv::float16_t, cv::float16_t, float, fop, init)

#define CV_IMPLEMENT_LOGIC_OP_ALLTYPES(name, op) \
    CV_IMPLEMENT_BINARY_OP(name, 8u, uint8_t, uint8_t, uint8_t, op, noinit) \
    CV_IMPLEMENT_BINARY_OP(name, 16u, uint16_t, uint16_t, uint16_t, op, noinit) \
    CV_IMPLEMENT_BINARY_OP(name, 32u, uint32_t, uint32_t, uint32_t, op, noinit) \
    CV_IMPLEMENT_BINARY_OP(name, 64u, uint64_t, uint64_t, uint64_t, op, noinit)

#define CV_IMPLEMENT_CMP_ALLTYPES(name, op) \
    CV_IMPLEMENT_BINARY_OP(name, 8u, uint8_t, bool, uint8_t, op, noinit) \
    CV_IMPLEMENT_BINARY_OP(name, 8s, int8_t, bool, int8_t, op, noinit) \
    CV_IMPLEMENT_BINARY_OP(name, 16u, uint16_t, bool, uint16_t, op, noinit) \
    CV_IMPLEMENT_BINARY_OP(name, 16s, int16_t, bool, int16_t, op, noinit) \
    CV_IMPLEMENT_BINARY_OP(name, 32u, uint32_t, bool, uint32_t, op, noinit) \
    CV_IMPLEMENT_BINARY_OP(name, 32s, int32_t, bool, int32_t, op, noinit) \
    CV_IMPLEMENT_BINARY_OP(name, 64u, uint64_t, bool, uint64_t, op, noinit) \
    CV_IMPLEMENT_BINARY_OP(name, 64s, int64_t, bool, int64_t, op, noinit) \
    CV_IMPLEMENT_BINARY_OP(name, 32f, float, bool, float, op, noinit) \
    CV_IMPLEMENT_BINARY_OP(name, 16f, cv::float16_t, bool, float, op, noinit)

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
#undef CV_MEAN
#define CV_MEAN(x, y) (((x) + (y))*scale)
#define noinit

CV_IMPLEMENT_ARITHM_OP_ALLTYPES(add, CV_ADD, CV_ADD, noinit)
CV_IMPLEMENT_ARITHM_OP_ALLTYPES(sub, CV_SUB, CV_SUB, noinit)
CV_IMPLEMENT_ARITHM_OP_ALLTYPES(mul, CV_MUL, CV_MUL, noinit)
CV_IMPLEMENT_ARITHM_OP_ALLTYPES(div, CV_DIV, CV_DIV, noinit)
CV_IMPLEMENT_ARITHM_OP_ALLTYPES(mod, CV_MOD, fmodf, noinit)
CV_IMPLEMENT_ARITHM_OP_ALLTYPES(max, std::max, std::max, noinit)
CV_IMPLEMENT_ARITHM_OP_ALLTYPES(min, std::min, std::min, noinit)
CV_IMPLEMENT_ARITHM_OP_ALLTYPES(pow, std::pow, powf, noinit)
CV_IMPLEMENT_ARITHM_OP_ALLTYPES(mean, CV_MEAN, CV_MEAN, float scale=params[0])

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

ElemwiseOp::forward_t ElemwiseOp::getForwardSlice(ElemwiseOpcode opcode, int type)
{
#undef CV_INIT_ARITHM_FUNC_TAB
#define CV_INIT_ARITHM_FUNC_TAB(name) \
    static forward_t name##_tab[CV_DEPTH_MAX] = { \
        elemwise_##name##_8u, elemwise_##name##_8s, elemwise_##name##_16u, \
        elemwise_##name##_16s, elemwise_##name##_32s, elemwise_##name##_32f, \
        0, elemwise_##name##_16f, 0, 0, elemwise_##name##_64u, \
        elemwise_##name##_64s, elemwise_##name##_32u }

#undef CV_INIT_LOGIC_FUNC_TAB
#define CV_INIT_LOGIC_FUNC_TAB(name) \
    static forward_t name##_tab[CV_DEPTH_MAX] = { \
        elemwise_##name##_8u, elemwise_##name##_8u, elemwise_##name##_16u, \
        elemwise_##name##_16u, elemwise_##name##_32u, \
        elemwise_##name##_32u, elemwise_##name##_64u, elemwise_##name##_16u, \
        elemwise_##name##_16u, elemwise_##name##_8u, \
        elemwise_##name##_64u, elemwise_##name##_64u, elemwise_##name##_32u }

#undef CV_INIT_MATH_FUNC_TAB
#define CV_INIT_MATH_FUNC_TAB(name) \
    static forward_t name##_tab[CV_DEPTH_MAX] = { \
        0, 0, 0, 0, 0, elemwise_##name##_32f, 0, elemwise_##name##_16f }

    static forward_t neg_tab[CV_DEPTH_MAX] = {
        0, elemwise_neg_8s, 0, elemwise_neg_16s, elemwise_neg_32s, elemwise_neg_32f, \
        0, elemwise_neg_16f, 0, 0, 0, elemwise_neg_64s };

    CV_INIT_ARITHM_FUNC_TAB(add);
    CV_INIT_ARITHM_FUNC_TAB(sub);
    CV_INIT_ARITHM_FUNC_TAB(mul);
    CV_INIT_ARITHM_FUNC_TAB(div);
    CV_INIT_ARITHM_FUNC_TAB(mod);
    CV_INIT_ARITHM_FUNC_TAB(max);
    CV_INIT_ARITHM_FUNC_TAB(min);
    CV_INIT_ARITHM_FUNC_TAB(pow);
    CV_INIT_ARITHM_FUNC_TAB(mean);
    CV_INIT_ARITHM_FUNC_TAB(eq);
    CV_INIT_ARITHM_FUNC_TAB(ne);
    CV_INIT_ARITHM_FUNC_TAB(lt);
    CV_INIT_ARITHM_FUNC_TAB(le);
    CV_INIT_ARITHM_FUNC_TAB(ge);
    CV_INIT_ARITHM_FUNC_TAB(gt);

    CV_INIT_LOGIC_FUNC_TAB(and);
    CV_INIT_LOGIC_FUNC_TAB(or);
    CV_INIT_LOGIC_FUNC_TAB(xor);
    CV_INIT_LOGIC_FUNC_TAB(not);

    CV_INIT_MATH_FUNC_TAB(abs);
    CV_INIT_MATH_FUNC_TAB(acos);
    CV_INIT_MATH_FUNC_TAB(acosh);
    CV_INIT_MATH_FUNC_TAB(asin);
    CV_INIT_MATH_FUNC_TAB(asinh);
    CV_INIT_MATH_FUNC_TAB(atan);
    CV_INIT_MATH_FUNC_TAB(atanh);
    CV_INIT_MATH_FUNC_TAB(ceil);
    CV_INIT_MATH_FUNC_TAB(cos);
    CV_INIT_MATH_FUNC_TAB(cosh);
    CV_INIT_MATH_FUNC_TAB(erf);
    CV_INIT_MATH_FUNC_TAB(exp);
    CV_INIT_MATH_FUNC_TAB(floor);
    CV_INIT_MATH_FUNC_TAB(log);
    CV_INIT_MATH_FUNC_TAB(relu);
    CV_INIT_MATH_FUNC_TAB(round);
    CV_INIT_MATH_FUNC_TAB(sigmoid);
    CV_INIT_MATH_FUNC_TAB(sign);
    CV_INIT_MATH_FUNC_TAB(sin);
    CV_INIT_MATH_FUNC_TAB(sinh);
    CV_INIT_MATH_FUNC_TAB(sqrt);
    CV_INIT_MATH_FUNC_TAB(tan);
    CV_INIT_MATH_FUNC_TAB(tanh);

    CV_INIT_MATH_FUNC_TAB(clip);
    CV_INIT_MATH_FUNC_TAB(leaky_relu);

    static volatile bool initialized = false;
    static forward_t* func_tabs[ELWISE_OPCODE_MAX];

    if (!initialized) {
        func_tabs[ELWISE_ADD] = add_tab;
        func_tabs[ELWISE_AND] = and_tab;
        func_tabs[ELWISE_DIV] = div_tab;
        func_tabs[ELWISE_EQUAL] = eq_tab;
        func_tabs[ELWISE_GREATER] = gt_tab;
        func_tabs[ELWISE_GREATER_EQUAL] = ge_tab;
        func_tabs[ELWISE_LESS] = lt_tab;
        func_tabs[ELWISE_LESS_EQUAL] = le_tab;
        func_tabs[ELWISE_MAX] = max_tab;
        func_tabs[ELWISE_MEAN] = add_tab;
        func_tabs[ELWISE_MIN] = min_tab;
        func_tabs[ELWISE_MOD] = mod_tab;
        func_tabs[ELWISE_MUL] = mul_tab;
        func_tabs[ELWISE_POW] = pow_tab;
        func_tabs[ELWISE_OR] = or_tab;
        func_tabs[ELWISE_SUB] = sub_tab;
        func_tabs[ELWISE_XOR] = xor_tab;

        func_tabs[ELWISE_ABS] = abs_tab;
        func_tabs[ELWISE_ACOS] = acos_tab;
        func_tabs[ELWISE_ACOSH] = acosh_tab;
        func_tabs[ELWISE_ASIN] = asin_tab;
        func_tabs[ELWISE_ASINH] = asinh_tab;
        func_tabs[ELWISE_ATAN] = atan_tab;
        func_tabs[ELWISE_ATANH] = atanh_tab;
        func_tabs[ELWISE_CEIL] = ceil_tab;
        func_tabs[ELWISE_CLIP] = clip_tab;
        func_tabs[ELWISE_COS] = cos_tab;
        func_tabs[ELWISE_COSH] = cosh_tab;
        func_tabs[ELWISE_ERF] = erf_tab;
        func_tabs[ELWISE_EXP] = exp_tab;
        func_tabs[ELWISE_FLOOR] = floor_tab;
        func_tabs[ELWISE_ISINF] = nullptr;
        func_tabs[ELWISE_ISNAN] = nullptr;
        func_tabs[ELWISE_LOG] = log_tab;
        func_tabs[ELWISE_LRELU] = leaky_relu_tab;
        //func_tabs[ELWISE_MISH] = mish_tab;
        func_tabs[ELWISE_NEG] = neg_tab;
        func_tabs[ELWISE_NOT] = not_tab;
        func_tabs[ELWISE_RELU] = relu_tab;
        func_tabs[ELWISE_ROUND] = round_tab;
        func_tabs[ELWISE_SIGMOID] = sigmoid_tab;
        func_tabs[ELWISE_SIGN] = sign_tab;
        func_tabs[ELWISE_SIN] = sin_tab;
        func_tabs[ELWISE_SINH] = sinh_tab;
        func_tabs[ELWISE_SOFTPLUS] = nullptr;
        func_tabs[ELWISE_SOFTSIGN] = nullptr;
        func_tabs[ELWISE_SQRT] = sqrt_tab;
        func_tabs[ELWISE_TAN] = tan_tab;
        func_tabs[ELWISE_TANH] = tanh_tab;
        initialized = true;
    }
    forward_t* tab_ptr = CV_MAT_CN(type) == 1 ? func_tabs[opcode] : nullptr;
    return tab_ptr ? tab_ptr[CV_MAT_DEPTH(type)] : nullptr;
}

ElemwiseOp::forward_t ElemwiseOp::getForwardSlice(int type) const
{
    return getForwardSlice(opcode, type);
}

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

    virtual void forward(Net2& net, Graph& graph,
                        const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs,
                        std::vector<Buffer>& tempbufs) CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());

        AutoBuffer<int64_t> sizesbuf((ninputs+1)*2*(sizeof(TensorSize)/sizeof(int64_t)) + (ninputs + 1)*(2 + TensorSize::MAX_NDIMS));
        TensorSize* sizes0 = (TensorSize*)sizesbuf.data(), *sizes = sizes0 + ninputs + 1;
        size_t** steps = (size_t**)(sizes + ninputs + 1);
        void** dataptr0 = (void**)(steps + ninputs + 1);
        size_t* stepsdata = (size_t*)(dataptr0 + ninputs + 1);

        int inptype0 = inputs[0].type(), outtype = inferType(inptype0);
        TensorSize* outsize = &sizes0[ninputs];
        sizes0[0] = *outsize = inputs[0].size();
        steps[0] = stepsdata;
        steps[ninputs] = stepsdata + ninputs*TensorSize::MAX_NDIMS;
        CV_Assert(inputs[0].isContinuous());
        dataptr0[0] = (void*)inputs[0].data();

        for (size_t k = 1; k < ninputs; k++) {
            sizes0[k] = inputs[k].size();
            CV_Assert(inputs[k].type() == inputs[0].type());
            CV_Assert(inputs[0].isContinuous());
            *outsize = outsize->expand(sizes0[k]);
            steps[k] = stepsdata + k*TensorSize::MAX_NDIMS;
            dataptr0[k] = (void*)inputs[k].data();
        }

        forward_t funcptr = getForwardSlice(inptype0);
        forward_t addptr = opcode == ELWISE_MEAN && ninputs > 2 ?
            getForwardSlice(ELWISE_ADD, inptype0) : nullptr;
        CV_Assert(funcptr != nullptr);

        outputs.resize(1);
        Tensor& out = outputs[0];
        out.fitSameDevice(inputs[0], *outsize, outtype);
        CV_Assert(out.isContinuous());

        dataptr0[ninputs] = (void*)out.data();

        // some of inputs are empty => result is empty
        int max_ndims = prepareForBroadcasting(3, sizes0, sizes, steps);
        if (max_ndims <= 0)
            return;
        outsize = &sizes[ninputs];
        int64_t nslices = 1, slicesize = outsize->size[max_ndims-1];
        for (int i = 0; i < max_ndims-1; i++)
            nslices *= outsize->size[i];

        size_t inpesz = CV_ELEM_SIZE(inptype0);
        size_t outesz = CV_ELEM_SIZE(outtype);

        if (nslices == 1) {
            int64_t BLOCK_SIZE = ninputs == 1 ? 1<<14 : 1<<17;
            int64_t nblocks = (slicesize + BLOCK_SIZE-1)/BLOCK_SIZE;
            parallel_for_(Range(0, (int)nblocks), [&](const Range& r) {
                int64_t start = r.start*BLOCK_SIZE/nblocks;
                int64_t end = std::min(r.end*BLOCK_SIZE/nblocks, slicesize);
                const void* inptr[2];
                void* outptr;
                size_t curr_steps[3] = {steps[0][max_ndims-1], 1, 1};
                float scale = 0.;
                const float* curr_params = params;
                if (opcode == ELWISE_MEAN) {
                    scale = 1.f/ninputs;
                    curr_params = &scale;
                }

                inptr[0] = (char*)dataptr0[0] + inpesz*curr_steps[0]*start;
                outptr = (char*)dataptr0[ninputs] + outesz*start;

                if (ninputs == 1) {
                    funcptr(1, inptr, curr_steps, outptr, (size_t)(end - start), curr_params);
                } else {
                    for (size_t k = 1; k < ninputs; k++) {
                        curr_steps[1] = steps[k][max_ndims-1];
                        inptr[1] = (char*)dataptr0[k] + inpesz*curr_steps[1]*start;
                        if (k+1 == ninputs || opcode != ELWISE_MEAN)
                            funcptr(2, inptr, curr_steps, outptr, (size_t)(end - start), curr_params);
                        else
                            addptr(2, inptr, curr_steps, outptr, (size_t)(end - start), curr_params);
                        inptr[0] = outptr;
                        curr_steps[0] = 1;
                    }
                }
            });
        } else {
            CV_Assert(ninputs > 1);
            const void* inptr[2];
            void* outptr;
            int64_t idxbuf[TensorSize::MAX_NDIMS];
            float scale = 0.;
            const float* curr_params = params;
            if (opcode == ELWISE_MEAN) {
                scale = 1.f/ninputs;
                curr_params = &scale;
            }

            for (int64_t slice_idx = 0; slice_idx < nslices; slice_idx++) {
                size_t curr_steps[3] = {steps[0][max_ndims-1], 1, 1};
                int64_t start1 = 0, start = 0;
                int64_t idx = slice_idx;
                for (int i = max_ndims-2; i >= 0; i--) {
                    int64_t size_i = outsize->size[i];
                    int64_t superslice_idx = idx/size_i;
                    int64_t i_k = idx - superslice_idx*size_i;
                    idxbuf[i] = i_k;
                    start1 += i_k*steps[0][i];
                    start += i_k*steps[ninputs][i];
                    idx = superslice_idx;
                }

                inptr[0] = (char*)dataptr0[0] + inpesz*start1;
                outptr = (char*)dataptr0[ninputs] + outesz*start;

                for (size_t k = 1; k < ninputs; k++) {
                    curr_steps[1] = steps[k][max_ndims-1];
                    start1 = 0;
                    for (int i = 0; i <= max_ndims-2; i++)
                        start1 += idxbuf[i]*steps[k][i];
                    inptr[1] = (char*)dataptr0[k] + inpesz*start1;
                    if (k+1 == ninputs || opcode != ELWISE_MEAN)
                        funcptr(2, inptr, curr_steps, outptr, (size_t)slicesize, curr_params);
                    else
                        addptr(2, inptr, curr_steps, outptr, (size_t)slicesize, curr_params);
                    inptr[0] = outptr;
                    curr_steps[0] = 1;
                }
            }
        }
    }

    bool haveFP16;
};

Op ElemwiseOp::create(ElemwiseOpcode opcode,
                      const float* params, size_t nparams)
{
    return std::make_shared<ElemwiseOpImpl>(opcode, params, nparams);
}

}}
