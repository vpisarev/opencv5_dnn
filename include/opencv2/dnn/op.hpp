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
    ELWISE_CLIPC,
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
    ELWISE_TANH
};

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
    REDUCE_SUM_SQUARE
};

CV_EXPORTS Arg constant(PGraph& graph, InputArray arr);
CV_EXPORTS Arg constScalar(PGraph& graph, int value);
CV_EXPORTS Arg constScalar(PGraph& graph, int64_t value);
CV_EXPORTS Arg constScalar(PGraph& graph, double value);
CV_EXPORTS Arg constVector(PGraph& graph, const std::vector<int>& values);
CV_EXPORTS Arg constVector(PGraph& graph, const std::vector<int64_t>& values);
CV_EXPORTS Arg constVector(PGraph& graph, const std::vector<double>& values);

struct CV_EXPORTS ElemwiseOp : public BaseOp
{
public:
    virtual ~ElemwiseOp();
    virtual ElemwiseOpcode elemwiseOpcode() const;

    virtual void forwardSlice(int ninputs, const float** inputs, float* output, size_t len) const;
    virtual void forwardSlice(int ninputs, const cv::float16_t** inputs, cv::float16_t* output, size_t len) const;
    virtual void forwardSlice(int ninputs, const cv::bfloat16_t** inputs, cv::bfloat16_t* output, size_t len) const;

    double param1, param2;
};

CV_EXPORTS Arg elemwise(PGraph& graph, ElemwiseOpcode opcode, Arg input);
CV_EXPORTS Arg elemwise(PGraph& graph, ElemwiseOpcode opcode, Arg input0, Arg input1);
CV_EXPORTS Arg elemwise(PGraph& graph, ElemwiseOpcode opcode, const std::vector<Arg>& inputs);

CV_EXPORTS Arg add(PGraph& graph, Arg input0, Arg input1);
CV_EXPORTS Arg bitwise_and(PGraph& graph, Arg input0, Arg input1);
CV_EXPORTS Arg divide(PGraph& graph, Arg input0, Arg input1);
CV_EXPORTS Arg equal(PGraph& graph, Arg input0, Arg input1);
CV_EXPORTS Arg greater(PGraph& graph, Arg input0, Arg input1);
CV_EXPORTS Arg greaterEqual(PGraph& graph, Arg input0, Arg input1);
CV_EXPORTS Arg less(PGraph& graph, Arg input0, Arg input1);
CV_EXPORTS Arg lessEqual(PGraph& graph, Arg input0, Arg input1);

CV_EXPORTS Arg max(PGraph& graph, Arg input0, Arg input1);
CV_EXPORTS Arg max(PGraph& graph, const std::vector<Arg>& inputs);
CV_EXPORTS Arg mean(PGraph& graph, Arg input0, Arg input1);
CV_EXPORTS Arg mean(PGraph& graph, const std::vector<Arg>& inputs);
CV_EXPORTS Arg min(PGraph& graph, Arg input0, Arg input1);
CV_EXPORTS Arg min(PGraph& graph, const std::vector<Arg>& inputs);

CV_EXPORTS Arg mod(PGraph& graph, Arg input0, Arg input1);
CV_EXPORTS Arg multiply(PGraph& graph, Arg input0, Arg input1);
CV_EXPORTS Arg pow(PGraph& graph, Arg input0, Arg input1);
CV_EXPORTS Arg bitwise_or(PGraph& graph, Arg input0, Arg input1);
CV_EXPORTS Arg subtract(PGraph& graph, Arg input0, Arg input1);
CV_EXPORTS Arg bitwise_xor(PGraph& graph, Arg input0, Arg input1);

CV_EXPORTS Arg abs(PGraph& graph, Arg input);
CV_EXPORTS Arg acos(PGraph& graph, Arg input);
CV_EXPORTS Arg acosh(PGraph& graph, Arg input);
CV_EXPORTS Arg asin(PGraph& graph, Arg input);
CV_EXPORTS Arg asinh(PGraph& graph, Arg input);
CV_EXPORTS Arg atan(PGraph& graph, Arg input);
CV_EXPORTS Arg atanh(PGraph& graph, Arg input);
CV_EXPORTS Arg ceil(PGraph& graph, Arg input);
CV_EXPORTS Arg clip(PGraph& graph, Arg input, Arg minval, Arg maxval);
CV_EXPORTS Arg cos(PGraph& graph, Arg input);
CV_EXPORTS Arg cosh(PGraph& graph, Arg input);
CV_EXPORTS Arg erf(PGraph& graph, Arg input);
CV_EXPORTS Arg exp(PGraph& graph, Arg input);
CV_EXPORTS Arg floor(PGraph& graph, Arg input);
CV_EXPORTS Arg isinf(PGraph& graph, Arg input);
CV_EXPORTS Arg isnan(PGraph& graph, Arg input);
CV_EXPORTS Arg log(PGraph& graph, Arg input);
CV_EXPORTS Arg leakyRelu(PGraph& graph, Arg input, double alpha);
CV_EXPORTS Arg mish(PGraph& graph, Arg input);
CV_EXPORTS Arg neg(PGraph& graph, Arg input);
CV_EXPORTS Arg bitwise_not(PGraph& graph, Arg input);
CV_EXPORTS Arg relu(PGraph& graph, Arg input);
CV_EXPORTS Arg round(PGraph& graph, Arg input);
CV_EXPORTS Arg sigmoid(PGraph& graph, Arg input);
CV_EXPORTS Arg sign(PGraph& graph, Arg input);
CV_EXPORTS Arg sin(PGraph& graph, Arg input);
CV_EXPORTS Arg sinh(PGraph& graph, Arg input);
CV_EXPORTS Arg softplus(PGraph& graph, Arg input);
CV_EXPORTS Arg softsign(PGraph& graph, Arg input);
CV_EXPORTS Arg sqrt(PGraph& graph, Arg input);
CV_EXPORTS Arg tan(PGraph& graph, Arg input);
CV_EXPORTS Arg tanh(PGraph& graph, Arg input);

struct CV_EXPORTS ReduceOp : public BaseOp
{
public:
    virtual ~ReduceOp();
    virtual ReduceOpcode reduceOpcode() const;
};


CV_EXPORTS Arg reduce(PGraph& graph, ReduceOpcode opcode, Arg input, Arg axes,
                      bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceL1(PGraph& graph, Arg input, Arg axes,
                        bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceL2(PGraph& graph, Arg input, Arg axes,
                        bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceLogSum(PGraph& graph, Arg input, Arg axes,
                            bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceLogSumExp(PGraph& graph, Arg input, Arg axes,
                               bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceMax(PGraph& graph, Arg input, Arg axes,
                         bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceMean(PGraph& graph, Arg input, Arg axes,
                         bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceMin(PGraph& graph, Arg input, Arg axes,
                         bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceProd(PGraph& graph, Arg input, Arg axes,
                          bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceSum(PGraph& graph, Arg input, Arg axes,
                         bool keepdims=true, bool noOpWithEmptyAxes=false);
CV_EXPORTS Arg reduceSumSquare(PGraph& graph, Arg input, Arg axes,
                               bool keepdims=true, bool noOpWithEmptyAxes=false);

struct CV_EXPORTS ConvParams
{
    std::vector<int> size;
    std::vector<int> stride={};
    std::vector<int> dilation={};
    std::vector<int> padding={};
};

/*
    Arg Max
*/
struct CV_EXPORTS ArgMaxOp : public BaseOp
{
    virtual ~ArgMaxOp();

    int axis;
    bool keepdims;
    bool selectLastIndex;
};

CV_EXPORTS Arg argMax(PGraph& graph, Arg input, int axis=0, bool keepdims=true, bool selectLastIndex=false);

/*
    Arg Min
*/
struct CV_EXPORTS ArgMinOp : public BaseOp
{
    virtual ~ArgMinOp();

    int axis;
    bool keepdims;
    bool selectLastIndex;
};

CV_EXPORTS Arg argMin(PGraph& graph, Arg input, int axis=0, bool keepdims=true, bool selectLastIndex=false);


/*
    Average Pooling
*/
struct CV_EXPORTS AveragePoolOp : public BaseOp
{
    virtual ~AveragePoolOp();

    ConvParams params;
    bool countIncludePadding;
};

CV_EXPORTS Arg averagePool(PGraph& graph, Arg input, const ConvParams& params, bool countIncludePadding=false);


/*
    Batch Normalization
*/
struct CV_EXPORTS BatchNormOp : public BaseOp
{
    virtual ~BatchNormOp();

    double epsilon;
    double momentum;
    bool trainingMode;
};

CV_EXPORTS Arg batchNorm(PGraph& graph, Arg input, Arg scale, Arg B, Arg mean, Arg variance);

/*
    Type cast
*/
struct CV_EXPORTS CastOp : public BaseOp
{
    virtual ~CastOp();

    int type;
};

CV_EXPORTS Arg cast(PGraph& graph, Arg input, int type);

/*
    Concatenate several tensors into one
*/
struct CV_EXPORTS ConcatOp : public BaseOp
{
    virtual ~ConcatOp();

    int axis;
};

CV_EXPORTS Arg concat(PGraph& graph, const std::vector<Arg>& inputs, int axis);

/*
    Constant of shape
*/
struct CV_EXPORTS ConstantOfShapeOp : public BaseOp
{
    virtual ~ConstantOfShapeOp();

    Tensor value;
};

CV_EXPORTS Arg constantOfShape(PGraph& graph, Arg shape, InputArray value);

/*
    Convolution
*/
struct CV_EXPORTS ConvOp : public BaseOp
{
    virtual ~ConvOp();

    ConvParams params;
    Op batchNorm; // fused batch norm, if any
    Op activ; // fused activation, if any
    bool fused_residual;
};

CV_EXPORTS Arg conv(PGraph& graph, Arg input, Arg weights, const ConvParams& params);

/*
    Transposed Convolution
*/
struct CV_EXPORTS ConvTransposeOp : public BaseOp
{
    virtual ~ConvTransposeOp();

    ConvParams params;
};

CV_EXPORTS Arg convTranspose(PGraph& graph, Arg input, Arg weights, const ConvParams& params);

/*
    Dropout
*/
struct CV_EXPORTS DropoutOp : public BaseOp
{
    virtual ~DropoutOp();

    int64_t seed;
};

CV_EXPORTS Arg dropout(PGraph& graph, Arg input, Arg ratio, Arg trainingMode);

/*
    Expand
*/
struct CV_EXPORTS ExpandOp : public BaseOp
{
    virtual ~ExpandOp();
};

CV_EXPORTS Arg expand(PGraph& graph, Arg input, Arg shape);

/*
    Flatten
*/
struct CV_EXPORTS FlattenOp : public BaseOp
{
    virtual ~FlattenOp();

    int axis;
};

CV_EXPORTS Arg flatten(PGraph& graph, Arg input);

/*
    Gather
*/
struct CV_EXPORTS GatherOp : public BaseOp
{
    virtual ~GatherOp();

    int axis;
};

CV_EXPORTS Arg gather(PGraph& graph, Arg input, Arg ind);


/*
    Gemm
*/
struct CV_EXPORTS GemmOp : public BaseOp
{
    virtual ~GemmOp();

    double alpha, beta;
    bool transA, transB;
};

CV_EXPORTS Arg gemm(PGraph& graph, Arg A, Arg B, Arg bias);


/*
    Global Average Pooling
*/
struct CV_EXPORTS GlobalAveragePoolOp : public BaseOp
{
    virtual ~GlobalAveragePoolOp();
};

CV_EXPORTS Arg globalAveragePool(PGraph& graph, Arg input);


/*
    Identity
*/
struct CV_EXPORTS IdentityOp : public BaseOp
{
    virtual ~IdentityOp();
};

CV_EXPORTS Arg identity(PGraph& graph, Arg input);


/*
    If
*/
struct CV_EXPORTS IfOp : public BaseOp
{
    virtual ~IfOp();
};

CV_EXPORTS void if_(PGraph& graph, Arg input, const PGraph& thenGraph,
                    const PGraph& elseGraph, std::vector<Arg>& out);

/*
    Instance Normalization
*/
struct CV_EXPORTS InstanceNormalizeOp : public BaseOp
{
    virtual ~InstanceNormalizeOp();

    double epsilon;
};

CV_EXPORTS Arg instanceNormalize(PGraph& graph, Arg input, Arg scale, Arg bias, double epsilon=1e-5);

/*
    Layer Normalization
*/
struct CV_EXPORTS LayerNormalizeOp : public BaseOp
{
    virtual ~LayerNormalizeOp();

    int axis;
    double epsilon;
    int stashType;
};

CV_EXPORTS Arg layerNormalize(PGraph& graph, Arg input, Arg scale, Arg bias, int axis=-1,
                              double epsilon=1e-5, int stashType=CV_32F);

/*
    Loop
*/
struct CV_EXPORTS LoopOp : public BaseOp
{
    virtual ~LoopOp();
    size_t noutputs;
};

CV_EXPORTS void loop(PGraph& graph, const PGraph& body, Arg tripCount, Arg CondIn,
                    const std::vector<Arg>& inputs, size_t noutputs, std::vector<Arg>& outputs);

/*
    LRN
*/
struct CV_EXPORTS LRNOp : public BaseOp
{
    virtual ~LRNOp();
};

CV_EXPORTS Arg LRN(PGraph& graph, Arg input, int size, double alpha=0.0001, double beta=0.75, double bias=1.0);

/*
    Matrix Multiplication
*/
struct CV_EXPORTS MatMulOp : public BaseOp
{
    virtual ~MatMulOp();
};

CV_EXPORTS Arg matMul(PGraph& graph, Arg A, Arg B);


/*
    Max Pooling
*/
struct CV_EXPORTS MaxPoolOp : public BaseOp
{
    virtual ~MaxPoolOp();

    ConvParams params;
    bool computeIndices;
    bool rowMajorOrder;
};

CV_EXPORTS Arg maxPool(PGraph& graph, Arg input, const ConvParams& params,
                       bool computeIndices=false, bool rowMajorOrder=true);

/*
    Non-max Suppression
*/
struct CV_EXPORTS NonMaxSuppressionOp : public BaseOp
{
    virtual ~NonMaxSuppressionOp();

    ConvParams params;
    bool centerPointBox;
};

CV_EXPORTS Arg nonMaxSuppression(PGraph& graph, Arg boxes, Arg scores, Arg maxOutputBoxesPerClass,
                                 Arg iouThreshold, Arg scoreThreshold, bool centerPointBox=false);

/*
    Non zero
*/
struct CV_EXPORTS NonZeroOp : public BaseOp
{
    virtual ~NonZeroOp();
};

CV_EXPORTS Arg nonZero(PGraph& graph, Arg input);


/*
    Range
*/
struct CV_EXPORTS RangeOp : public BaseOp
{
    virtual ~RangeOp();
};

CV_EXPORTS Arg range(PGraph& graph, Arg start, Arg limit, Arg delta);

/*
    Reshape
*/
struct CV_EXPORTS ReshapeOp : public BaseOp
{
    virtual ~ReshapeOp();

    bool allowZero;
};

CV_EXPORTS Arg reshape(PGraph& graph, Arg input, Arg shape, bool allowZero=false);


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
    virtual ~ResizeOp();

    ResizeParams params;
};

CV_EXPORTS Arg resize(PGraph& graph, Arg input, Arg scales, Arg sizes, Arg roi, const ResizeParams& params);


/*
    Scatter
*/
struct CV_EXPORTS ScatterOp : public BaseOp
{
    virtual ~ScatterOp();

    int axis;
};

CV_EXPORTS Arg scatter(PGraph& graph, Arg data, Arg updates, Arg indices, int axis);


/*
    Shape
*/
struct CV_EXPORTS ShapeOp : public BaseOp
{
    virtual ~ShapeOp();

    int start, end;
};

CV_EXPORTS Arg shape(PGraph& graph, Arg input, int start=0, int end=INT_MAX);


/*
    Slice
*/
struct CV_EXPORTS SliceOp : public BaseOp
{
    virtual ~SliceOp();

    int start, end;
};

CV_EXPORTS Arg slice(PGraph& graph, Arg input, int start=0, int end=INT_MAX);


/*
    SoftMax
*/
struct CV_EXPORTS SoftMaxOp : public BaseOp
{
    virtual ~SoftMaxOp();

    int start, end;
};

CV_EXPORTS Arg softMax(PGraph& graph, Arg input, int start=0, int end=INT_MAX);


/*
    Split
*/
struct CV_EXPORTS SplitOp : public BaseOp
{
    virtual ~SplitOp();

    size_t noutputs;
};

CV_EXPORTS Arg split(PGraph& graph, Arg input, Arg split, size_t noutputs=0);


/*
    Squeeze
*/
struct CV_EXPORTS SqueezeOp : public BaseOp
{
    virtual ~SqueezeOp();
};

CV_EXPORTS Arg squeeze(PGraph& graph, Arg input, Arg axes);


/*
    Tile
*/
struct CV_EXPORTS TileOp : public BaseOp
{
    virtual ~TileOp();
};

CV_EXPORTS Arg tile(PGraph& graph, Arg input, Arg repeats);


/*
    TopK
*/
struct CV_EXPORTS TopKOp : public BaseOp
{
    virtual ~TopKOp();

    int axis;
    bool largest;
    bool sorted;
};

CV_EXPORTS Arg topK(PGraph& graph, Arg input, int axis=-1, bool largest=true, bool sorted=true);

/*
    Transpose
*/
struct CV_EXPORTS TransposeOp : public BaseOp
{
    virtual ~TransposeOp();

    std::vector<int> perm;
};

CV_EXPORTS Arg transpose(PGraph& graph, Arg input, const std::vector<int>& perm);


/*
    Unsqueeze
*/
struct CV_EXPORTS UnsqueezeOp : public BaseOp
{
    virtual ~UnsqueezeOp();
};

CV_EXPORTS Arg unsqueeze(PGraph& graph, Arg input, Arg axes);

}}

#endif
