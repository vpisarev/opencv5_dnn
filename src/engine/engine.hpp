// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_ENGINE_HPP__
#define __OPENCV_DNN_ENGINE_HPP__

#include "opencv2/dnn2.hpp"

namespace cv { namespace dnn {

typedef std::pair<int64_t, std::string> OnnxOpSet;

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
    OnnxArgInfo() : typ(-1) {}
    std::string name;
    int typ;
    std::vector<OnnxTensorDim> shape;
};

struct OnnxInfo
{
    int64_t IRVersion;
    std::string producer;
    std::string domain;
    std::string docString;
    std::vector<OnnxOpSet> opsets;
};

typedef std::unordered_map<std::string, int> NamesHash;
typedef std::unordered_map<std::string, double> profile;

struct Net2::Impl
{
    Impl(Net2* net_);
    ~Impl();

    void clear();
    void initArgs();
    void forwardGraph(const Graph& graph);
    void useCounts(std::vector<int>& usecounts) const;
    void updateUseCounts(std::vector<int>& usecounts, const Graph& graph) const;

    void initProfile();
    void updateProfile(const Op& op);

    void checkArgs(const std::vector<Arg>& args) const;
    void checkArg(Arg arg) const;

    void assignBuffers();
    void fuse();
    void foldConstSubexpr();

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
    TracingMode tracingMode;
    ProfilingMode profilingMode;
    int accuracy;

    Buffer scratchBuf;
    std::vector<int64_t> perfProfileTime;
    std::vector<int> perfProfileCount;

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

// normalize axis. The input axis should be within [-ndims, ndims-1] range
int normalizeAxis(int axis, int ndims);

// extract array of axes (up to TensorSize::MAX_DIMS).
// Each axis should be within a range [-ndims, ndims-1]
int normalizeAxes(const Tensor& axes, int ndims, int* axisbuf, bool* axismask=nullptr);

// computes shape of the output tensor of convolution
// (including depth-wise convolution), max pooling or average pooling operations
TensorSize convInferShape(const TensorSize& inpsize, const ConvParams& convparams);

struct DepthwiseConvParams
{
    int64_t KH, KW, SY, SX, DY, DX;
    int64_t pad_y0, pad_x0, pad_y1, pad_x1;
    int64_t N, Hi, Wi, H, W, C1, C0;
    int64_t inner_y0, inner_x0, inner_y1, inner_x1;
    const int* yxtab;
    const int64_t* ofstab;

    std::ostream& dump(std::ostream& strm);
};

// initializes the structure of parameters for 1D/2D/3D
// depth-wise convolution, max pooling or average pooling
DepthwiseConvParams initDepthwiseConv(const TensorSize& inpsize,
                                      const ConvParams& convparams,
                                      int* yxtab, int64_t* ofstab);

void prindent(std::ostream& strm, int indent);

}}

#endif
