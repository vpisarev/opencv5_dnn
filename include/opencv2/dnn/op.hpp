// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_OP_HPP__
#define __OPENCV_DNN_OP_HPP__

namespace cv { namespace dnn {

enum ElemwiseOpcode
{
    ELWISE_NONE = 0,
    ELWISE_ADD,
    ELWISE_AND,
    ELWISE_DIV,
    ELWISE_EQUAL,
    ELWISE_GREATER,
    ELWISE_GREATER_EQUAL,
    ELWISE_LESS,
    ELWISE_LESS_EQUAL,
    ELWISE_MAX,
    ELWISE_MEAN,
    ELWISE_MIN,
    ELWISE_MOD,
    ELWISE_MUL,
    ELWISE_POW,
    ELWISE_OR,
    ELWISE_SUB,
    ELWISE_SUM,
    ELWISE_XOR,

    ELWISE_ABS,
    ELWISE_ACOS,
    ELWISE_ACOSH,
    ELWISE_ASIN,
    ELWISE_ASINH,
    ELWISE_ATAN,
    ELWISE_ATANH,
    ELWISE_CEIL,
    ELWISE_CLIP,
    ELWISE_COS,
    ELWISE_COSH,
    ELWISE_ERF,
    ELWISE_EXP,
    ELWISE_FLOOR,
    ELWISE_ISINF,
    ELWISE_ISNAN,
    ELWISE_LOG,
    ELWISE_LRELU,
    ELWISE_MISH,
    ELWISE_NEG,
    ELWISE_NOT,
    ELWISE_RELU,
    ELWISE_ROUND,
    ELWISE_SIGMOID,
    ELWISE_SIGN,
    ELWISE_SIN,
    ELWISE_SINH,
    ELWISE_SOFTPLUS,
    ELWISE_SOFTSIGN,
    ELWISE_SQRT,
    ELWISE_TAN,
    ELWISE_TANH,

    ELWISE_OPCODE_MAX,
};

CV_EXPORTS std::string_view elemwiseOpcode2str(ElemwiseOpcode opcode);

enum ReduceOpcode
{
    REDUCE_NONE=0,
    REDUCE_L1,
    REDUCE_L2,
    REDUCE_LOGSUM,
    REDUCE_LOGSUMEXP,
    REDUCE_MAX,
    REDUCE_MEAN,
    REDUCE_MIN,
    REDUCE_PROD,
    REDUCE_SUM,
    REDUCE_SUM_SQUARE,

    REDUCE_OPCODE_MAX
};

CV_EXPORTS std::string_view reduceOpcode2str(ReduceOpcode opcode);

CV_EXPORTS Arg constant(Graph& graph, std::string_view opname,
                        std::string_view outname, InputArray arr);
CV_EXPORTS Arg constScalar(Graph& graph, std::string_view opname,
                           std::string_view outname, int depth, const void* value);
template<typename _Tp> CV_INLINE Arg constScalar(Graph& graph, std::string_view opname,
                           std::string_view outname, _Tp value)
{
    return constScalar(graph, opname, outname, DataType<_Tp>::type, &value);
}
CV_EXPORTS Arg constVector(Graph& graph, std::string_view opname,
                           std::string_view outname, int type,
                           const void* data, size_t len);
template<typename _Tp> CV_INLINE Arg constVector(Graph& graph, std::string_view opname,
                           std::string_view outname, const std::vector<_Tp>& values)
{
    return constVector(graph, opname, outname, DataType<_Tp>::type,
                       values.data(), values.size());
}

struct CV_EXPORTS ElemwiseOp : public BaseOp
{
public:
    enum { MAX_PARAMS=10 };
    typedef void (*forward_t)(size_t ninputs, const void** inputs, const size_t* steps,
                              void* output, size_t len, const float* params);
    typedef void (*activ_t)(const void* input, void* output, size_t len, const float* params);
    static Op create(ElemwiseOpcode opcode, const float* params=nullptr, size_t nparams=0);
    virtual ~ElemwiseOp();

    static forward_t getForwardSlice(ElemwiseOpcode opcode, int type);
    static activ_t getActivation(ElemwiseOpcode opcode, int type);
    virtual forward_t getForwardSlice(int type) const;
    virtual activ_t getActivation(int type) const;
    ElemwiseOpcode opcode;
    float params[MAX_PARAMS];
};

CV_EXPORTS Arg elemwise(Graph& graph, std::string_view opname, std::string_view outname,
                        ElemwiseOpcode opcode, Arg input,
                        const float* params=nullptr, size_t nparams=0);
CV_EXPORTS Arg elemwise(Graph& graph, std::string_view opname, std::string_view outname,
                        ElemwiseOpcode opcode, Arg input0, Arg input1,
                        const float* params=nullptr, size_t nparams=0);
CV_EXPORTS Arg elemwise(Graph& graph, std::string_view opname, std::string_view outname,
                        ElemwiseOpcode opcode, const std::vector<Arg>& inputs,
                        const float* params=nullptr, size_t nparams=0);

CV_EXPORTS Arg add(Graph& graph, std::string_view opname,
                   std::string_view outname, Arg input0, Arg input1);
CV_EXPORTS Arg bitwise_and(Graph& graph, std::string_view opname,
                           std::string_view outname, Arg input0, Arg input1);
CV_EXPORTS Arg divide(Graph& graph, std::string_view opname,
                      std::string_view outname, Arg input0, Arg input1);
CV_EXPORTS Arg equal(Graph& graph, std::string_view opname,
                     std::string_view outname, Arg input0, Arg input1);
CV_EXPORTS Arg greater(Graph& graph, std::string_view opname,
                       std::string_view outname, Arg input0, Arg input1);
CV_EXPORTS Arg greaterEqual(Graph& graph, std::string_view opname,
                            std::string_view outname, Arg input0, Arg input1);
CV_EXPORTS Arg less(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input0, Arg input1);
CV_EXPORTS Arg lessEqual(Graph& graph, std::string_view opname,
                         std::string_view outname, Arg input0, Arg input1);

CV_EXPORTS Arg max(Graph& graph, std::string_view opname,
                   std::string_view outname, Arg input0, Arg input1);
CV_EXPORTS Arg max(Graph& graph, std::string_view opname,
                   std::string_view outname, const std::vector<Arg>& inputs);
CV_EXPORTS Arg mean(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input0, Arg input1);
CV_EXPORTS Arg mean(Graph& graph, std::string_view opname,
                    std::string_view outname, const std::vector<Arg>& inputs);
CV_EXPORTS Arg min(Graph& graph, std::string_view opname,
                   std::string_view outname, Arg input0, Arg input1);
CV_EXPORTS Arg min(Graph& graph, std::string_view opname,
                   std::string_view outname, const std::vector<Arg>& inputs);
CV_EXPORTS Arg sum(Graph& graph, std::string_view opname,
                   std::string_view outname, const std::vector<Arg>& inputs);

CV_EXPORTS Arg mod(Graph& graph, std::string_view opname,
                   std::string_view outname, Arg input0, Arg input1);
CV_EXPORTS Arg multiply(Graph& graph, std::string_view opname,
                        std::string_view outname, Arg input0, Arg input1);
CV_EXPORTS Arg pow(Graph& graph, std::string_view opname,
                   std::string_view outname, Arg input0, Arg input1);
CV_EXPORTS Arg bitwise_or(Graph& graph, std::string_view opname,
                          std::string_view outname, Arg input0, Arg input1);
CV_EXPORTS Arg subtract(Graph& graph, std::string_view opname,
                        std::string_view outname, Arg input0, Arg input1);
CV_EXPORTS Arg bitwise_xor(Graph& graph, std::string_view opname,
                           std::string_view outname, Arg input0, Arg input1);

CV_EXPORTS Arg abs(Graph& graph, std::string_view opname,
                   std::string_view outname, Arg input);
CV_EXPORTS Arg acos(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input);
CV_EXPORTS Arg acosh(Graph& graph, std::string_view opname,
                     std::string_view outname, Arg input);
CV_EXPORTS Arg asin(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input);
CV_EXPORTS Arg asinh(Graph& graph, std::string_view opname,
                     std::string_view outname, Arg input);
CV_EXPORTS Arg atan(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input);
CV_EXPORTS Arg atanh(Graph& graph, std::string_view opname,
                     std::string_view outname, Arg input);
CV_EXPORTS Arg ceil(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input);
CV_EXPORTS Arg clip(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input, Arg minval, Arg maxval);
CV_EXPORTS Arg cos(Graph& graph, std::string_view opname,
                   std::string_view outname, Arg input);
CV_EXPORTS Arg cosh(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input);
CV_EXPORTS Arg erf(Graph& graph, std::string_view opname,
                   std::string_view outname, Arg input);
CV_EXPORTS Arg exp(Graph& graph, std::string_view opname,
                   std::string_view outname, Arg input);
CV_EXPORTS Arg floor(Graph& graph, std::string_view opname,
                     std::string_view outname, Arg input);
CV_EXPORTS Arg isinf(Graph& graph, std::string_view opname,
                     std::string_view outname, Arg input);
CV_EXPORTS Arg isnan(Graph& graph, std::string_view opname,
                     std::string_view outname, Arg input);
CV_EXPORTS Arg log(Graph& graph, std::string_view opname,
                   std::string_view outname, Arg input);
CV_EXPORTS Arg leakyRelu(Graph& graph, std::string_view opname,
                         std::string_view outname, Arg input, double alpha);
CV_EXPORTS Arg mish(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input);
CV_EXPORTS Arg neg(Graph& graph, std::string_view opname,
                   std::string_view outname, Arg input);
CV_EXPORTS Arg bitwise_not(Graph& graph, std::string_view opname,
                           std::string_view outname, Arg input);
CV_EXPORTS Arg relu(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input);
CV_EXPORTS Arg round(Graph& graph, std::string_view opname,
                     std::string_view outname, Arg input);
CV_EXPORTS Arg sigmoid(Graph& graph, std::string_view opname,
                       std::string_view outname, Arg input);
CV_EXPORTS Arg sign(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input);
CV_EXPORTS Arg sin(Graph& graph, std::string_view opname,
                   std::string_view outname, Arg input);
CV_EXPORTS Arg sinh(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input);
CV_EXPORTS Arg softplus(Graph& graph, std::string_view opname,
                        std::string_view outname, Arg input);
CV_EXPORTS Arg softsign(Graph& graph, std::string_view opname,
                        std::string_view outname, Arg input);
CV_EXPORTS Arg sqrt(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input);
CV_EXPORTS Arg tan(Graph& graph, std::string_view opname,
                   std::string_view outname, Arg input);
CV_EXPORTS Arg tanh(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input);

struct CV_EXPORTS ReduceOp : public BaseOp
{
public:
    static Op create(ReduceOpcode opcode, bool keepdims=true,
                     bool noOpWithEmptyAxes=false);
    virtual ~ReduceOp();
    ReduceOpcode opcode;
    bool keepdims;
    bool noOpWithEmptyAxes;
};


CV_EXPORTS Arg reduce(Graph& graph, std::string_view opname, std::string_view outname,
                      ReduceOpcode opcode, Arg input, Arg axes,
                      bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceL1(Graph& graph, std::string_view opname,
                        std::string_view outname, Arg input, Arg axes,
                        bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceL2(Graph& graph, std::string_view opname,
                        std::string_view outname, Arg input, Arg axes,
                        bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceLogSum(Graph& graph, std::string_view opname,
                            std::string_view outname, Arg input, Arg axes,
                            bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceLogSumExp(Graph& graph, std::string_view opname,
                               std::string_view outname, Arg input, Arg axes,
                               bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceMax(Graph& graph, std::string_view opname,
                         std::string_view outname, Arg input, Arg axes,
                         bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceMean(Graph& graph, std::string_view opname,
                          std::string_view outname, Arg input, Arg axes,
                         bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceMin(Graph& graph, std::string_view opname,
                         std::string_view outname, Arg input, Arg axes,
                         bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceProd(Graph& graph, std::string_view opname,
                          std::string_view outname, Arg input, Arg axes,
                          bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceSum(Graph& graph, std::string_view opname,
                         std::string_view outname, Arg input, Arg axes,
                         bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceSumSquare(Graph& graph, std::string_view opname,
                               std::string_view outname, Arg input, Arg axes,
                               bool keepdims=true, bool noOpWithEmptyAxes=false);

struct CV_EXPORTS ConvParams
{
    std::vector<int> ksizes;
    std::vector<int> strides={};
    std::vector<int> dilations={};
    std::vector<int> pads={};
    int ngroups = 0;
    std::ostream& dump(std::ostream& strm);
};

/*
    Arg Max
*/
struct CV_EXPORTS ArgMaxOp : public BaseOp
{
    static Op create(int axis, bool keepdims, bool selectLastIndex);
    virtual ~ArgMaxOp();

    int axis;
    bool keepdims;
    bool selectLastIndex;
};

CV_EXPORTS Arg argMax(Graph& graph, std::string_view opname,
                      std::string_view outname, Arg input,
                      int axis=0, bool keepdims=true, bool selectLastIndex=false);

/*
    Arg Min
*/
struct CV_EXPORTS ArgMinOp : public BaseOp
{
    static Op create(int axis, bool keepdims, bool selectLastIndex);
    virtual ~ArgMinOp();

    int axis;
    bool keepdims;
    bool selectLastIndex;
};

CV_EXPORTS Arg argMin(Graph& graph, std::string_view opname,
                      std::string_view outname, Arg input,
                      int axis=0, bool keepdims=true, bool selectLastIndex=false);


/*
    Average Pooling
*/
struct CV_EXPORTS AveragePoolOp : public BaseOp
{
    static Op create(const ConvParams& convparams, bool countIncludePadding);
    virtual ~AveragePoolOp();

    ConvParams params;
    bool countIncludePadding;
};

CV_EXPORTS Arg averagePool(Graph& graph, std::string_view opname,
                           std::string_view outname, Arg input,
                           const ConvParams& params, bool countIncludePadding=false);


/*
    Batch Normalization
*/
struct CV_EXPORTS BatchNormOp : public BaseOp
{
    static Op create(double epsilon=1e-5);
    virtual ~BatchNormOp();
    virtual void computeScaleBias(const Tensor& scale0, const Tensor& bias0,
                                  const Tensor& mean, const Tensor& variance) = 0;
    Tensor scale, bias;
    double epsilon;
};

CV_EXPORTS Arg batchNorm(Graph& graph, std::string_view opname,
                         std::string_view outname,
                         Arg input, Arg scale, Arg B, Arg mean,
                         Arg variance, double epsilon=1e-5);

/*
    Type cast
*/
struct CV_EXPORTS CastOp : public BaseOp
{
    static Op create(int type);
    virtual ~CastOp();

    int type;
};

CV_EXPORTS Arg cast(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input, int type);

/*
    Concatenate several tensors into one
*/
struct CV_EXPORTS ConcatOp : public BaseOp
{
    static Op create(int axis);
    virtual ~ConcatOp();

    int axis;
};

CV_EXPORTS Arg concat(Graph& graph, std::string_view opname, std::string_view outname,
                      const std::vector<Arg>& inputs, int axis);

/*
    Constant of shape
*/
struct CV_EXPORTS ConstantOfShapeOp : public BaseOp
{
    static Op create(Tensor value);
    virtual ~ConstantOfShapeOp();

    Tensor value;
};

CV_EXPORTS Arg constantOfShape(Graph& graph, std::string_view opname,
                               std::string_view outname,
                               Arg shape, InputArray value);

/*
    Convolution
*/
struct CV_EXPORTS ConvOp : public BaseOp
{
    static Op create(const ConvParams& convparams);
    virtual ~ConvOp();

    virtual void setWeights(const Tensor& weights, const Tensor& bias, int64_t C0, int accuracy=-1) = 0;
    virtual void fuseBatchNorm(const Op& batchNorm) = 0;
    virtual void fuseActivation(const Op& activ) = 0;

    ConvParams params;
    Op batchNorm; // fused batch norm, if any
    Op activ; // fused activation, if any
};

CV_EXPORTS Arg conv(Graph& graph, std::string_view opname, std::string_view outname,
                    Arg input, Arg weights, Arg bias, const ConvParams& params);

/*
    Transposed Convolution
*/
struct CV_EXPORTS ConvTransposeOp : public BaseOp
{
    static Op create(const ConvParams& convparams);
    virtual ~ConvTransposeOp();

    ConvParams params;
};

CV_EXPORTS Arg convTranspose(Graph& graph, std::string_view opname, std::string_view outname,
                             Arg input, Arg weights, Arg bias, const ConvParams& params);

/*
    Dropout
*/
struct CV_EXPORTS DropoutOp : public BaseOp
{
    static Op create(int64_t seed);
    virtual ~DropoutOp();

    int64_t seed;
};

CV_EXPORTS Arg dropout(Graph& graph, std::string_view opname, std::string_view outname,
                       Arg input, Arg ratio, Arg trainingMode);

/*
    Expand
*/
struct CV_EXPORTS ExpandOp : public BaseOp
{
    static Op create();
    virtual ~ExpandOp();
};

CV_EXPORTS Arg expand(Graph& graph, std::string_view opname,
                      std::string_view outname, Arg input, Arg shape);

/*
    Flatten
*/
struct CV_EXPORTS FlattenOp : public BaseOp
{
    static Op create(int axis=1);
    virtual ~FlattenOp();

    int axis;
};

CV_EXPORTS Arg flatten(Graph& graph, std::string_view opname,
                       std::string_view outname, Arg input, int axis=1);

/*
    Gather
*/
struct CV_EXPORTS GatherOp : public BaseOp
{
    static Op create(int axis);
    virtual ~GatherOp();

    int axis;
};

CV_EXPORTS Arg gather(Graph& graph, std::string_view opname,
                      std::string_view outname, Arg input, Arg ind);


/*
    Gemm
*/
struct CV_EXPORTS GemmOp : public BaseOp
{
    static Op create(double alpha, double beta, bool transA, bool transB);
    virtual ~GemmOp();

    double alpha, beta;
    bool transA, transB;
};

CV_EXPORTS Arg gemm(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg A, Arg B, Arg bias);


/*
    Global Average Pooling
*/
struct CV_EXPORTS GlobalAveragePoolOp : public BaseOp
{
    static Op create();
    virtual ~GlobalAveragePoolOp();
};

CV_EXPORTS Arg globalAveragePool(Graph& graph, std::string_view opname,
                                 std::string_view outname, Arg input);


/*
    Identity
*/
struct CV_EXPORTS IdentityOp : public BaseOp
{
    static Op create();
    virtual ~IdentityOp();
};

CV_EXPORTS Arg identity(Graph& graph, std::string_view opname,
                        std::string_view outname, Arg input);

/*
    If
*/
struct CV_EXPORTS IfOp : public BaseOp
{
    static Op create();
    virtual ~IfOp();
};

CV_EXPORTS void if_(Graph& graph, std::string_view opname,
                    const std::vector<std::string>& outnames,
                    Arg input, const Graph& thenGraph,
                    const Graph& elseGraph, std::vector<Arg>& out);

/*
    Instance Normalization
*/
struct CV_EXPORTS InstanceNormalizeOp : public BaseOp
{
    static Op create(double epsilon);
    virtual ~InstanceNormalizeOp();

    double epsilon;
};

CV_EXPORTS Arg instanceNormalize(Graph& graph, std::string_view opname,
                                 std::string_view outname, Arg input,
                                 Arg scale, Arg bias, double epsilon=1e-5);

/*
    Layer Normalization
*/
struct CV_EXPORTS LayerNormalizeOp : public BaseOp
{
    static Op create(int axis, double epsilon, int stashType);
    virtual ~LayerNormalizeOp();

    int axis;
    double epsilon;
    int stashType;
};

CV_EXPORTS Arg layerNormalize(Graph& graph, std::string_view opname,
                              std::string_view outname, Arg input, Arg scale, Arg bias,
                              int axis=-1, double epsilon=1e-5, int stashType=CV_32F);

/*
    Loop
*/
struct CV_EXPORTS LoopOp : public BaseOp
{
    static Op create(size_t noutputs);
    virtual ~LoopOp();
    size_t noutputs;
};

CV_EXPORTS void loop(Graph& graph, std::string_view opname,
                     const std::vector<std::string>& outnames,
                     const Graph& body, Arg tripCount, Arg CondIn,
                     const std::vector<Arg>& inputs,
                     size_t noutputs, std::vector<Arg>& outputs);

/*
    LRN
*/
struct CV_EXPORTS LRNOp : public BaseOp
{
    static Op create(double alpha, double beta, double bias, int size);
    virtual ~LRNOp();

    double alpha, beta, bias;
    int size;
};

CV_EXPORTS Arg LRN(Graph& graph, std::string_view opname,
                   std::string_view outname, Arg input, int size,
                   double alpha=0.0001, double beta=0.75, double bias=1.0);

/*
    Matrix Multiplication
*/
struct CV_EXPORTS MatMulOp : public BaseOp
{
    virtual ~MatMulOp();
};

CV_EXPORTS Arg matMul(Graph& graph, std::string_view opname,
                      std::string_view outname, Arg A, Arg B);

/*
    Max Pooling
*/
struct CV_EXPORTS MaxPoolOp : public BaseOp
{
    static Op create(const ConvParams& convparams, bool computeIndices, bool rowMajorOrder);
    virtual ~MaxPoolOp();

    ConvParams params;
    bool computeIndices;
    bool rowMajorOrder;
};

CV_EXPORTS Arg maxPool(Graph& graph, std::string_view opname,
                       std::string_view outname, Arg input, const ConvParams& params,
                       bool computeIndices=false, bool rowMajorOrder=true);

/*
    Non-max Suppression
*/
struct CV_EXPORTS NonMaxSuppressionOp : public BaseOp
{
    static Op create(bool centerPointBox);
    virtual ~NonMaxSuppressionOp();

    bool centerPointBox;
};

CV_EXPORTS Arg nonMaxSuppression(Graph& graph, std::string_view opname,
                                 std::string_view outname, Arg boxes, Arg scores,
                                 Arg maxOutputBoxesPerClass, Arg iouThreshold,
                                 Arg scoreThreshold, bool centerPointBox=false);

/*
    Non zero
*/
struct CV_EXPORTS NonZeroOp : public BaseOp
{
    static Op create();
    virtual ~NonZeroOp();
};

CV_EXPORTS Arg nonZero(Graph& graph, std::string_view opname,
                       std::string_view outname, Arg input);

/*
    Padding
*/
struct CV_EXPORTS PadOp : public BaseOp
{
    static Op create(int borderMode);
    virtual ~PadOp();

    int borderMode;
};

CV_EXPORTS Arg pad(Graph& graph, std::string_view opname,
                   std::string_view outname, Arg input, Arg paddings,
                   Arg axes=Arg(), int borderMode=BORDER_CONSTANT, Arg borderValue=Arg());

/*
    Range
*/
struct CV_EXPORTS RangeOp : public BaseOp
{
    static Op create();
    virtual ~RangeOp();
};

CV_EXPORTS Arg range(Graph& graph, std::string_view opname,
                     std::string_view outname, Arg start, Arg limit, Arg delta);

/*
    Reshape
*/
struct CV_EXPORTS ReshapeOp : public BaseOp
{
    static Op create(bool allowZero);
    virtual ~ReshapeOp();

    bool allowZero;
};

CV_EXPORTS Arg reshape(Graph& graph, std::string_view opname,
                       std::string_view outname, Arg input,
                       Arg shape, bool allowZero=false);


enum CoordTransMode
{
    CT_HalfPixel = 0,
    CT_PyTorchHalfPixel,
    CT_AlignCorners,
    CT_Asymmetric,
    CT_TFCropResize,
    CT_OutHalfPixel
};

enum InterpolationMode
{
    Inter_Nearest=0,
    Inter_Linear,
    Inter_Cubic
};

enum NearestNeighborMode
{
    Nearest_RoundPreferFloor=0,
    Nearest_RoundPreferCeil,
    Nearest_Floor,
    Nearest_Ceil
};

struct ResizeParams
{
    CoordTransMode coordTrans = CT_HalfPixel;
    InterpolationMode interpolation = Inter_Nearest;
    NearestNeighborMode nearest = Nearest_RoundPreferFloor;
    double cubicCoeffA = -0.75;
    bool excludeOutside = false;
    double extrapolationValue = 0.;
};

/*
    Reshape
*/
struct CV_EXPORTS ResizeOp : public BaseOp
{
    static Op create(const ResizeParams& params);
    virtual ~ResizeOp();

    ResizeParams params;
};

CV_EXPORTS Arg resize(Graph& graph, std::string_view opname, std::string_view outname,
                      Arg input, Arg scales, Arg sizes, Arg roi, const ResizeParams& params);


/*
    Scatter
*/
struct CV_EXPORTS ScatterOp : public BaseOp
{
    static Op create(int axis);
    virtual ~ScatterOp();

    int axis;
};

CV_EXPORTS Arg scatter(Graph& graph, std::string_view opname, std::string_view outname,
                       Arg data, Arg updates, Arg indices, int axis);


/*
    Shape
*/
struct CV_EXPORTS ShapeOp : public BaseOp
{
    static Op create(int start, int end);
    virtual ~ShapeOp();

    int start, end;
};

CV_EXPORTS Arg shape(Graph& graph, std::string_view opname, std::string_view outname,
                     Arg input, int start=0, int end=INT_MAX);


/*
    Slice
*/
struct CV_EXPORTS SliceOp : public BaseOp
{
    static Op create();
    virtual ~SliceOp();
};

CV_EXPORTS Arg slice(Graph& graph, std::string_view opname, std::string_view outname,
                     Arg input, Arg starts, Arg ends, Arg axes=Arg(), Arg steps=Arg());

/*
    SoftMax
*/
struct CV_EXPORTS SoftMaxOp : public BaseOp
{
    static Op create(int axis);
    virtual ~SoftMaxOp();

    int axis;
};

CV_EXPORTS Arg softMax(Graph& graph, std::string_view opname,
                       std::string_view outname, Arg input, int axis);


/*
    Split
*/
struct CV_EXPORTS SplitOp : public BaseOp
{
    static Op create(int axis, size_t noutputs);
    virtual ~SplitOp();

    int axis;
    size_t noutputs;
};

CV_EXPORTS Arg split(Graph& graph, std::string_view opname, std::string_view outname,
                     Arg input, Arg split, int axis, size_t noutputs);


/*
    Squeeze
*/
struct CV_EXPORTS SqueezeOp : public BaseOp
{
    static Op create();
    virtual ~SqueezeOp();
};

CV_EXPORTS Arg squeeze(Graph& graph, std::string_view opname,
                       std::string_view outname, Arg input, Arg axes=Arg());


/*
    Tile
*/
struct CV_EXPORTS TileOp : public BaseOp
{
    static Op create();
    virtual ~TileOp();
};

CV_EXPORTS Arg tile(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input, Arg repeats);


/*
    TopK
*/
struct CV_EXPORTS TopKOp : public BaseOp
{
    static Op create(int axis, bool largest, bool sorted);
    virtual ~TopKOp();

    int axis;
    bool largest;
    bool sorted;
};

CV_EXPORTS Arg topK(Graph& graph, std::string_view opname,
                    std::string_view outname, Arg input, Arg K,
                    int axis=-1, bool largest=true, bool sorted=true);

/*
    TransformLayout
*/
struct CV_EXPORTS TransformLayoutOp : public BaseOp
{
    static Op create(TensorLayout layout, int64_t C0=0);
    virtual ~TransformLayoutOp();

    TensorLayout layout;
    int64_t C0;
};

CV_EXPORTS Arg transformLayout(Graph& graph, std::string_view opname,
                               std::string_view outname, Arg input,
                               TensorLayout layout, int64_t C0=0);

/*
    Transpose
*/
struct CV_EXPORTS TransposeOp : public BaseOp
{
    static Op create(const std::vector<int>& perm);
    virtual ~TransposeOp();

    std::vector<int> perm;
};

CV_EXPORTS Arg transpose(Graph& graph, std::string_view opname, std::string_view outname,
                         Arg input, const std::vector<int>& perm);


/*
    Unsqueeze
*/
struct CV_EXPORTS UnsqueezeOp : public BaseOp
{
    static Op create();
    virtual ~UnsqueezeOp();
};

CV_EXPORTS Arg unsqueeze(Graph& graph, std::string_view opname,
                         std::string_view outname, Arg input, Arg axes);

}}

#endif
