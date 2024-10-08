// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_ENGINE_HPP__
#define __OPENCV_DNN_ENGINE_HPP__

#include "opencv2/dnn2.hpp"

namespace cv { namespace dnn {

struct OnnxTensorDim
{
    OnnxTensorDim() : value(-1) {}
    explicit OnnxTensorDim(const std::string& p) : param(p), value(-1) {}
    explicit OnnxTensorDim(const char* p) : param(p), value(-1) {}
    explicit OnnxTensorDim(int64_t v) : param(), value(v) {}
    bool empty() { return param.empty() && value <= 0; }
    std::string param;
    int64_t value;
};

struct OnnxArgInfo
{
    OnnxArgInfo() : type(-1) {}
    std::string name;
    int type;
    std::vector<OnnxTensorDim> size;
};

typedef std::unordered_map<std::string, int64_t> NamesHash;
typedef std::unordered_map<std::string, double> profile;

struct Net2::Impl
{
    Impl(Net2* net_);
    ~Impl();

    void clear();
    void prepareForInference();
    void forwardGraph(Graph& graph, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);
    void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);
    void updateUseCounts(const Graph& graph, std::vector<int>& usecounts) const;
    void useCounts(std::vector<int>& usecounts) const;

    void initProfile();
    void updateProfile(const Op& op);

    void checkArgs(const std::vector<Arg>& args) const;
    void checkArg(Arg arg) const;
    void checkAndUpdateDim(const Graph& g, const Node& node, Arg inp, int j, int64_t value);

    void traceArg(std::ostream& strm_, const char* prefix, size_t i, Arg arg, bool dumpdata);
    Op fromBlock;
    Tensor fromBlockResult;

    void inferTypes();
    void inferShapes(bool symbolic);
    void assignBuffers();
    void useBlockLayout();
    void fuse();
    void constFold();
    void constArgs();

    Net2* net;
    ModelFormat modelFormat;
    OnnxInfo onnxInfo;
    int onnx_opset;

    NamesHash argnames;
    NamesHash dimnames;
    std::vector<std::string> dimnames_;
    std::vector<ArgInfo> args;
    std::vector<Tensor> tensors;
    std::vector<int> bufidxs;
    std::vector<Buffer> buffers;
    std::vector<Buffer> scratch_bufs;
    std::vector<ArgInfo> pattern_args;
    std::vector<Tensor> pattern_tensors;
    Graph mainGraph;

    NamesHash profileEntries;
    std::vector<double> profileTimes;
    std::vector<std::string> profileNames;
    std::vector<int> profileRuns;

    TensorLayout defaultLayout;
    bool enableFP16;
    bool haveFP16;
    bool prepared;
    TracingMode tracingMode;
    ProfilingMode profilingMode;
    int accuracy;

    Buffer scratchBuf;
    std::vector<int64_t> perfProfileTime;
    std::vector<int> perfProfileCount;
    std::vector<int64_t> dimvalues;

    Device* defaultDevice;
    MemoryManager* defaultMemoryManager;
    std::vector<GraphBackend*> backends;
    std::vector<std::vector<Buffer> > backendBufs;

    std::ostream* strm;
    int dump_indent;
};

int prepareForBroadcasting(int ntensors, const TensorSize* sizes0,
                           TensorSize* sizes, size_t** steps);

bool isIntType(int type);
bool isSignedIntType(int type);
bool isFPType(int type);

std::string typeToString(int type);
std::string argKindToString(ArgKind kind);

// normalize axis. The input axis should be within [-ndims, ndims-1] range
int normalizeAxis(int axis, int ndims);

// extract array of axes (up to TensorSize::MAX_DIMS).
// Each axis should be within a range [-ndims, ndims-1]
int normalizeAxes(const Tensor& axes, int ndims, int* axisbuf, bool* axismask=nullptr);

// computes shape of the output tensor of convolution
// (including depth-wise convolution), max pooling or average pooling operations
TensorSize convInferShape(Net2& net, const TensorSize& inpsize,
                          const ConvParams& convparams,
                          const TensorSize& wsize=TensorSize(),
                          bool symbolic=false);

enum FastActivation {
    ACTIV_NONE=0,
    ACTIV_RELU,
    ACTIV_LEAKY_RELU,
    ACTIV_CLIP
};

struct ConvState
{
    int64_t ngroups, K1;
    int64_t Hk, Wk, SY, SX, DY, DX;
    int64_t pad_y0, pad_x0, pad_y1, pad_x1;
    int64_t N, Hi, Wi, H, W, C1, C0, C;
    int64_t inner_y0, inner_x0, inner_y1, inner_x1;
    const int* yxtab;
    const int64_t* ofstab;

    FastActivation fastActivation;
    float activParams[ElemwiseOp::MAX_PARAMS];
    ElemwiseOp::activ_t activation;

    std::ostream& dump(std::ostream& strm);
    bool sameShape(const ConvState& cs) const;
};

// initializes the structure of parameters for 1D/2D/3D
// depth-wise convolution, max pooling or average pooling
ConvState initPoolingState(Net2& net, const TensorSize& inpsize,
                           const ConvParams& convparams,
                           int* yxtab, int64_t* ofstab);
ConvState initConvState(Net2& net, const TensorSize& inpsize,
                        const TensorSize& wsize,
                        const ConvParams& convparams,
                        const Op& activOp,
                        int* yxtab=nullptr, int64_t* ofstab=nullptr);

void prindent(std::ostream& strm, int indent);

typedef void (*depthwise_conv2d_t)(const void* inp__, const void* residual__,
                                   void* out__, const ConvState& cs,
                                   const void* weights__,
                                   const float* scale__,
                                   const float* bias__);

depthwise_conv2d_t getDepthwiseConv2DFunc(int depth);

void repackDepthwiseConvWeights(const void* inpw, int inptype,
                                void* outw, int outtype,
                                const TensorSize& wsize, int64_t C0);

}}

#endif
