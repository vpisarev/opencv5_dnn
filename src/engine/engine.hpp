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

enum {
    DNN_MODEL_GENERIC = 0,
    DNN_MODEL_ONNX = 1,
    DNN_MODEL_TF = 2
};

typedef std::unordered_map<std::string, int> NamesHash;

struct Net2::Impl
{
    Impl();
    ~Impl();

    void clear();
    void forward(InputArrayOfArrays inputBlobs, OutputArrayOfArrays outputBlobs);
    void forwardGraph(const PGraph& graph);
    void useCounts(std::vector<int>& usecounts) const;
    void updateUseCounts(std::vector<int>& usecounts, const PGraph& graph) const;

    Arg addConstTensor(const std::string& name, const Tensor& t, int idx=-1);
    Arg addArg(ArgKind argkind, const ArgInfo& arginfo);
    int64_t findDim(const std::string& dimname);
    Arg findArg(const std::string& argname);
    Arg findOutputArg(const std::string& argname);
    bool isConst(Arg arg) const;
    int kind(Arg arg) const;
    bool empty() const;
    bool useFP16() const;
    void set(int propId, double value);
    double get(int propId) const;
    void getTensors(const int* firstarg, size_t nargs, std::vector<Mat>& inputs) const;

    //template<typename _LayerType> bool isOp(const Node* node) const
    //{ return node && dynamic_cast<_LayerType>(*node->op.get()) != 0; }

    void assignBuffers();
    void fuse();
    void foldConstSubexpr();

    Net2* net;
    int modelFormat;
    OnnxInfo onnxInfo;

    NamesHash argnames;
    NamesHash dimnames;
    std::vector<std::string> dimnames_;
    std::vector<ArgInfo> args;
    std::vector<Tensor> tensors;
    std::vector<int> bufidxs;
    std::vector<Buffer> buffers;
    PGraph graph;
    DataLayout2 defaultLayout;
    bool enableFP16;
    bool haveFP16;
    bool trace;
    bool profile;
    bool traceProfile;

    Buffer scratchBuf;
    std::vector<int64_t> perfProfileTime;
    std::vector<int> perfProfileCount;
    std::string delta_indent = "   ";

    Device* defaultDevice;
    MemoryManager* defaultMemoryManager;
};

}}

#endif
