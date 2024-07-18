// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_DNN2_HPP
#define OPENCV_DNN_DNN2_HPP

#include <vector>
#include <opencv2/core.hpp>
#include "opencv2/core/async.hpp"
#include <opencv2/dnn.hpp>
#include <iostream>

namespace cv {
namespace dnn {
//! @addtogroup dnn
//! @{

void serial_for_(const Range& r, std::function<void (const Range&)> body, double nblocks=0);

enum TensorLayout
{
    LAYOUT_UNKNOWN = 0,
    LAYOUT_ND = 1,
    LAYOUT_NCHW = 2,
    LAYOUT_NHWC = 3,
    LAYOUT_NCHWc = 4
};

CV_EXPORTS std::string layoutToString(TensorLayout layout);

struct CV_EXPORTS TensorSize
{
    TensorSize();
    TensorSize(int ndims, const int64_t* size,
                TensorLayout layout=LAYOUT_UNKNOWN);
    TensorSize(std::initializer_list<int64_t> size,
                TensorLayout layout=LAYOUT_UNKNOWN);
    static TensorSize fromArray(InputArray m, TensorLayout layout=LAYOUT_UNKNOWN);
    int toMatShape(int* mshape, int maxdims) const;
    // convert to block layout; existing layout must be NCHW or NHWC
    TensorSize toBlock(int64_t C0) const;
    // convert from block layout.
    // the new layout must be explicitly specified and be NCHW or NHWC
    TensorSize fromBlock(TensorLayout newLayout) const;
    TensorSize expand(const TensorSize& another) const;
    bool haveSymbols() const;
    size_t total() const;
    bool empty() const;
    std::ostream& dump(std::ostream& strm) const;
    enum {MAX_DIMS=10};
    TensorLayout layout;
    int ndims;
    int64_t C;
    int64_t size[MAX_DIMS];
};

// when both 'layout' and 'another.layout' are block layouts, we also check 'C == another.C'
CV_EXPORTS bool operator == (const TensorSize& shape1, const TensorSize& shape2);
CV_EXPORTS bool operator != (const TensorSize& shape1, const TensorSize& shape2);

struct CV_EXPORTS SizeType
{
    TensorSize size;
    int type = 0;
    size_t totalBytes() const;
    SizeType toBlock(int64_t C0) const;
    SizeType fromBlock(TensorLayout layout) const;

    std::ostream& dump(std::ostream& strm) const;
};

CV_EXPORTS bool operator == (const SizeType& st0, const SizeType& st1);
CV_EXPORTS bool operator != (const SizeType& st0, const SizeType& st1);

enum BufAccess { DNN_BUF_READONLY=1, DNN_BUF_WRITEONLY=2, DNN_BUF_RW=3 };

struct MemoryManager;

enum DeviceType {
    Device_CPU=0,
    Device_IGPU=1,
    Device_DGPU=2,
    Device_NPU=3
};

struct CV_EXPORTS Device
{
    static Device* CPU();
    virtual ~Device();
    virtual DeviceType type() const = 0;
    virtual bool isCPU() const;
    virtual std::string_view name() const = 0;
    virtual bool supportType(int type) const = 0;
    virtual bool supportZeroCopy() const = 0;
    virtual int ndevices() const = 0;
    virtual int index() const = 0;
    virtual Device* getSameKindDevice(int index) const = 0;
    virtual MemoryManager* defaultMemoryManager() = 0;
    virtual bool isSameDevice(Device* device) const = 0;
};

struct CV_EXPORTS MemoryManager
{
    static MemoryManager* forCPU();
    virtual ~MemoryManager();
    virtual void* allocate(Device* device, size_t bufsize) = 0;
    virtual void deallocate(Device* device, void* handle) = 0;
    virtual void* map(Device* device, void* handle, size_t size, int access=DNN_BUF_RW) = 0;
    virtual void unmap(Device* device, void* handle, void* ptr, size_t size, int access=DNN_BUF_RW) = 0;
    virtual void copyFromDevice(Device* device, void* handle, size_t offset, void* dst, size_t size) = 0;
    virtual void copyToDevice(Device* device, const void* src, void* handle, size_t offset, size_t size) = 0;
    virtual void copyWithinDevice(Device* device, const void* srchandle, size_t srcoffset,
                                  void* dsthandle, size_t dstoffset, size_t size) = 0;
    virtual void fill(Device* device, void* handle, size_t offset, size_t nelems, const void* value, size_t vsize) = 0;
};

class BufferData;
typedef std::shared_ptr<BufferData> Buffer;

class CV_EXPORTS BufferData
{
public:
    BufferData();
    ~BufferData();

    static Buffer allocate(size_t size, Device* device=nullptr, MemoryManager* mm=nullptr);
    Buffer allocateOnSameDevice(size_t size) const;
    void fit(size_t size);
    void release();
    void* map(BufAccess access=DNN_BUF_RW);
    void unmap(BufAccess access=DNN_BUF_RW);

    Device* device() const;
    MemoryManager* memoryManager() const;
    void* handle() const;
    void* hostPtr() const;
    size_t size() const;

protected:
    Device* device_;
    MemoryManager* mm_;
    void* handle_;
    size_t size_;
    void* host_ptr_;
    int mapcount_;
};

// temporary alternative to the UMat (which is to be refactored)
class CV_EXPORTS Tensor
{
public:
    Tensor();
    Tensor(const Tensor& t);
    Tensor& operator = (const Tensor& t);
    ~Tensor();

    explicit Tensor(const TensorSize& size, int type, Device* device=nullptr);
    explicit Tensor(const TensorSize& size, int type, void* data, bool copy, Device* device=nullptr);
    explicit Tensor(const Buffer& buffer);
    explicit Tensor(const Buffer& buffer, size_t start, size_t maxsize);
    explicit Tensor(InputArray arr, TensorLayout layout, bool copy, Device* device=nullptr);

    static void multiFit(Buffer& buffer,
                        std::initializer_list<SizeType> st,
                        std::initializer_list<Tensor*> tensors,
                        size_t alignment=32);
    static Tensor makeScalar(int type, const void* value, Device* device=nullptr);
    template<typename _Tp> static Tensor makeScalar(_Tp value, Device* device=nullptr)
    {
        return makeScalar(DataType<_Tp>::type, &value, device);
    }
    static Tensor makeVector(int type, const void* value, size_t nelems, Device* device=nullptr);
    template<typename _Tp> static Tensor makeVector(const std::vector<_Tp>& values,
                                                    Device* device=nullptr)
    {
        return makeVector(DataType<_Tp>::type, values.data(), values.size(), device);
    }
    bool isScalar() const;
    bool getScalar(int type, void* scalar) const;
    template<typename _Tp> _Tp getScalar() const {
        _Tp value = 0;
        bool ok = getScalar(DataType<_Tp>::type, &value);
        return ok ? value : _Tp();
    }

    void release();
    void fit(const TensorSize& size, int type);
    void fitSameDevice(const Tensor& tensor, const TensorSize& size, int type);
    bool isOnSameDevice(const Tensor& tensor);
    bool samePlace(const Tensor& another) const;
    void setData(const TensorSize& size, int type, void* data, bool copy, Device* device);
    void setData(InputArray arr, TensorLayout layout, bool copy, Device* device);
    void setBuffer(const Buffer& buffer);
    void setBufferSlice(const Buffer& buffer, size_t start, size_t maxsize);
    Buffer buffer() const;

    bool isContinuous() const;
    bool usesBufferSlice() const;

    Device* device() const;
    MemoryManager* memoryManager() const;
    DeviceType deviceType() const;
    size_t total() const;
    size_t totalBytes() const;
    size_t elementSize() const;
    bool empty() const;
    // return pointer in device memory. When tensor is on the host, handle() == data().
    void* handle() const;
    size_t sliceStart() const;
    size_t sliceMaxSize() const;
    // return pointer in the host memory. When tensor is not on the host, exception is thrown.
    void* data() const;
    template<typename _Tp> _Tp* ptr() const { return (_Tp*)data(); }
    int ndims() const;
    TensorSize size() const;
    SizeType sizetype() const;
    TensorLayout layout() const;
    int type() const;
    int depth() const;
    int channels() const;

    Mat getMat() const;
    Tensor download() const;
    Tensor upload(Device* device) const;
    Tensor uploadToSameDevice(const Tensor& t) const;
    void copyTo(Tensor& tensor) const;
    void setTo(int type, const void* value);
    template<typename _Tp> void setTo(_Tp value) {
        setTo(DataType<_Tp>::type, &value);
    }
    void convertTo(Tensor& tensor, int type, double scale=1, double bias=0) const;
    Tensor reshape(const TensorSize& size) const;
    Tensor reinterpret(int type) const;
    template<typename _Tp> Tensor reinterpret() const
    { return reinterpret(DataType<_Tp>::type); }

    void* map(BufAccess access=DNN_BUF_RW);
    void unmap(BufAccess access=DNN_BUF_RW);

    std::ostream& dump(std::ostream& strm, int indent, int context=0,
              size_t maxsz_all=0, bool braces=true) const;

protected:
    enum { CONTINUOUS_FLAG=1, BUFFER_SLICE_FLAG=2 };
    void init();

    int flags_;
    int type_;
    TensorSize size_;
    Buffer buf_;
    void* ext_data_;
    size_t slice_start_;
    size_t slice_maxsize_;
};

class Net2;
struct NodeData;
struct GraphData;
struct GraphBackend;
struct BaseOp;

typedef std::shared_ptr<BaseOp> Op;
typedef std::shared_ptr<NodeData> Node;
typedef std::shared_ptr<GraphData> Graph;

struct CV_EXPORTS Arg
{
    Arg() : idx(0) {};
    explicit Arg(int idx_) : idx(idx_) {};
    bool empty() const { return idx == 0; }
    bool isPattern() const { return idx < 0; }
    // idx > 0: the Arg is input or output argument of some operation inside inference graph
    // idx < 0: the Arg is input or output argument of a pattern
    // idx == 0: no/empty argument; used in operations where some of the inputs/outputs are optional.
    int idx;
};

CV_EXPORTS bool operator == (const Arg& a, const Arg& b);

enum ArgKind { DNN_ARG_EMPTY=0, DNN_ARG_CONST, DNN_ARG_INPUT, DNN_ARG_OUTPUT, DNN_ARG_TEMP, DNN_ARG_PATTERN };

struct CV_EXPORTS ArgInfo
{
    ArgInfo();
    std::string name;
    ArgKind kind;
    TensorSize size;
    int type;
};

struct CV_EXPORTS BaseOp
{
public:
    virtual ~BaseOp();
    virtual std::string_view name() const = 0;
    virtual std::string_view origName() const;
    virtual std::string_view profileName() const;
    virtual std::ostream& dumpAttrs(std::ostream& strm, int indent) const;
    static void dumpTensorAttr(std::ostream& strm, std::string_view name,
                               const Tensor& t, int indent);
    static void dumpScalarAttr(std::ostream& strm, std::string_view name,
                               int type, const void* scalar, int indent);
    template<typename _Tp> static void dumpScalarAttr(std::ostream& strm,
                                                      std::string_view name, _Tp scalar, int indent)
    { dumpScalarAttr(strm, name, DataType<_Tp>::type, &scalar, indent); }
    static void dumpStringAttr(std::ostream& strm, std::string_view name,
                               std::string_view value, int indent);

    virtual Op clone() const = 0;
    virtual int minNumInputs() const = 0;
    virtual int maxNumInputs() const = 0;
    virtual int minNumOutputs() const = 0;
    virtual int maxNumOutputs() const = 0;

    virtual void setProfileEntry(int idx);
    virtual int getProfileEntry() const;

    virtual bool supportType(int input, int depth) const = 0;
    virtual bool alwaysSupportInplace() const;
    virtual bool supportInplace(const Net2& net, const Graph& graph,
                                const std::vector<Arg>& inpargs,
                                const std::vector<SizeType>& inpst) const;
    // 1 - yes (only block layout), 0 - support any layout, -1 - no (only non-block layout)
    virtual int supportBlockLayout(int input, int ninputs) const;

    virtual int64_t getFLOPS(const std::vector<SizeType> &inputs,
                           const std::vector<SizeType> &outputs) const;
    virtual void inferTypes(const Net2& net, const Graph& graph,
                            const std::vector<Arg>& inpargs,
                            const std::vector<int>& inptypes,
                            const std::vector<Arg>& outargs,
                            std::vector<int>& outtypes) const = 0;
    virtual void inferShapes(Net2& net, const Graph& graph,
                             const std::vector<Arg>& inpargs,
                             const std::vector<TensorSize>& inpshapes,
                             const std::vector<Arg>& outargs,
                             std::vector<TensorSize>& outshapes,
                             bool symbolic) const = 0;
    virtual void forward(Net2& net, Graph& graph,
                        const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs,
                        std::vector<Buffer>& tempbufs) = 0;
    virtual void forward(Net2& net, Graph& graph,
                        const std::vector<Tensor>& inputs,
                        Tensor& output,
                        std::vector<Buffer>& tempbufs);
protected:
    int profileIdx;
};

class CV_EXPORTS NodeData
{
public:
    NodeData();
    NodeData(const std::string_view name, const Op& op,
         const std::vector<Arg>& inputs,
         const std::vector<Arg>& outputs,
         const std::vector<Graph>& subgraphs=std::vector<Graph>());
    ~NodeData();
    static Node create(const std::string_view name, const Op& op,
                       const std::vector<Arg>& inputs,
                       const std::vector<Arg>& outputs,
                       const std::vector<Graph>& subgraphs=std::vector<Graph>());
    Node clone(Net2* newnet=nullptr) const;

    std::ostream& dump(const Net2& net, std::ostream& strm,
                       int indent, bool comma) const;

    std::string_view name() const;
    Op& op() const;
    const std::vector<Arg>& inputs() const;
    const std::vector<Arg>& outputs() const;
    const std::vector<Graph>& subgraphs() const;
    Arg inputs(size_t idx) const;
    Arg outputs(size_t idx) const;
    Graph subgraphs(size_t idx) const;
    size_t ninputs() const;
    size_t noutputs() const;
    size_t nsubgraphs() const;

protected:
    std::string name_;
    Op op_;
    std::vector<Arg> inputs_;
    std::vector<Arg> outputs_;
    std::vector<Graph> subgraphs_;
};

struct CV_EXPORTS BaseOptimizedGraph
{
    virtual ~BaseOptimizedGraph();
    virtual GraphBackend* getBackend() const = 0;
    virtual std::ostream& dump(const Net2& net, const Graph& g,
                               std::ostream& strm, int indent) const;
    virtual bool update(Net2& net, const Graph& g,
                        const std::vector<SizeType>& curr_inpst,
                        const std::vector<SizeType>& prev_inpst,
                        std::vector<Buffer>& tempbufs) const;
    virtual bool forward(Net2& net, const Graph& graph, std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs, std::vector<Buffer>& tempbufs) = 0;
};

typedef std::shared_ptr<BaseOptimizedGraph> OptimizedGraph;

class CV_EXPORTS GraphData
{
public:
    GraphData(const Net2& net, std::string_view name,
              const std::vector<Arg>& inputs,
              bool ispattern=false);
    ~GraphData();
    std::string_view name() const;
    bool empty() const;
    void clear();
    Graph clone(Net2* newnet=nullptr) const;
    void append(std::string_view node_name, const Op& op,
                const std::vector<std::string_view>& outnames,
                const std::vector<Arg>& inputs,
                std::vector<Arg>& outputs);
    Arg append(std::string_view node_name, const Op& op,
               std::string_view outname, const std::vector<Arg>& inputs);
    bool isPattern() const;
    void replaceAll(const std::vector<std::pair<Graph, Graph> >& subst);
    std::ostream& dump(std::ostream& strm, int indent, bool comma);
    void inferShapes(const std::vector<SizeType>& inpst,
                     std::vector<SizeType>& outst) const;
    Net2* net() const;
    const std::vector<Arg>& inputs() const;
    const std::vector<Arg>& outputs() const;
    void setOutputs(const std::vector<Arg>& outputs);
    const std::vector<Node>& prog() const;
    void setProg(const std::vector<Node>& newprog);
    OptimizedGraph getOptimized() const;
    void setOptimized(const OptimizedGraph& optigraph);

protected:
    bool ispattern_;
    Net2* net_;
    std::string_view name_;
    std::vector<Arg> inputs_;
    std::vector<Arg> outputs_;
    std::vector<Node> prog_;
    GraphBackend* backend_;
    OptimizedGraph optigraph_;
};

struct CV_EXPORTS GraphBackend
{
    static GraphBackend* CPU();
    static GraphBackend* fromSpec(std::string_view backendSpec);
    virtual ~GraphBackend();
    virtual Device* device() const = 0;
    virtual std::string_view name() const = 0;
    virtual bool supportType(int type) const = 0;
    virtual int64_t preferredBlockSize(int type) const = 0;
    virtual bool supportOp(const Op& op, const std::vector<SizeType>& inpst) const = 0;
    virtual void preprocessGraph(Graph& graph,
                        const std::vector<SizeType>& inpst,
                        std::vector<Buffer>& tempbufs) const;
    virtual bool forward(Graph& graph, std::vector<Tensor>& inputs,
                         std::vector<Tensor>& outputs,
                         std::vector<Buffer>& tempbufs) const;
};

enum TracingMode
{
    DNN_TRACE_NONE = 0,
    DNN_TRACE_ALL = 1,
    DNN_TRACE_OP = 2
};

enum ProfilingMode
{
    DNN_PROFILE_NONE = 0,
    DNN_PROFILE_SUMMARY = 1,
    DNN_PROFILE_DETAILED = 2
};

enum ModelFormat {
    DNN_MODEL_GENERIC = 0,
    DNN_MODEL_ONNX = 1,
    DNN_MODEL_TF = 2
};


typedef std::pair<int64_t, std::string> OnnxOpSet;
struct OnnxInfo
{
    int64_t IRVersion;
    std::string producer;
    std::string domain;
    std::string docString;
    std::vector<OnnxOpSet> opsets;
};

class CV_EXPORTS_W_SIMPLE Net2
{
public:
    Net2();  //!< Default constructor.
    ~Net2(); //!< Destructor frees the net only if there aren't references to the net anymore.

    void prepare();
    void release();
    void forward(InputArrayOfArrays inputBlobs,
                 OutputArrayOfArrays outputBlobs);
    void getInputNames(std::vector<std::string>& inputs) const;
    void getOutputNames(std::vector<std::string>& outputs) const;

    void setTracingMode(TracingMode mode);
    TracingMode getTracingMode() const;
    void setProfilingMode(ProfilingMode mode);
    ProfilingMode getProfilingMode() const;
    void getProfile(std::vector<std::string>& opnames, std::vector<double>& times) const;
    int registerProfileEntry(std::string_view opname);
    // mode 1: we know in advance names of all the graph inputs and outputs (e.g. when we parse ONNX).
    // The function register arguments with given names and
    // creates a new empty graph with the inputs and outputs.
    Graph newGraph(std::string_view name,
                   const std::vector<std::string>& inpnames,
                   const std::vector<std::string>& outnames) const;
    // mode 2: we construct the graph manually.
    // First, we create empty graph with certain input Args (they may or may not have names).
    // once the graph is constructed, we set the graph outputs using Graph::setOutputs().
    Graph newGraph(std::string_view name,
                   const std::vector<Arg>& inputs) const;
    Graph newPatternGraph(std::string_view name,
                          const std::vector<Arg>& inputs) const;
    Graph getMainGraph() const;
    void setMainGraph(const Graph& g);
    bool setAccuracy(int type); // typically, CV_16F/CV_16BF can be used to explicitly
                                // enable FP16/BF16 on backends that support both FP16/BF16 and FP32
    int getAccuracy() const;
    void checkArgs(const std::vector<Arg>& args) const;
    void checkArg(Arg arg) const;
    const ArgInfo& argInfo(Arg arg) const;
    std::string_view argName(Arg arg) const;
    ArgKind argKind(Arg arg) const;

    // if name is empty, always creates a new argument;
    // if it's not empty, returns argument with the specific name if it already exists,
    // otherwise creates new argument with the specified name
    Arg getArg(std::string_view name);
    bool haveArg(std::string_view name) const;

    Arg newConstArg(std::string_view name, const Tensor& t) const;
    Arg newConstScalarArg(std::string_view name, int type, const void* value) const;
    template<typename _Tp> Arg newConstScalarArg(std::string_view name, const _Tp& value) const
    {
        return newConstArg(name, Tensor::makeScalar(DataType<_Tp>::type, &value));
    }
    Arg newArg(std::string_view name, ArgKind kind) const;
    Arg newPatternArg() const;
    bool isConstArg(Arg arg) const;
    bool isTempArg(Arg arg) const;
    bool isPattern(Arg arg) const;
    Tensor argTensor(Arg arg) const;
    TensorSize argSize(Arg arg) const;
    int argType(Arg arg) const;
    SizeType argSizeType(Arg arg) const;

    int64_t findDim(std::string_view name=std::string_view(), bool insert=false);
    std::string dimToString(int64_t size) const;

    bool useBackend(std::string_view backendSpec); // CUDA:1, iGPU, NPU:0, ...
    bool useBackend(GraphBackend* backend); // the latest added backend gets the highest priority
    bool removeBackend(GraphBackend* backend);
    size_t getNumUsedBackends() const;
    GraphBackend* getBackend(size_t i) const; // 0 - highest-priority backend, ...

    Net2 clone() const;
    bool empty() const;

    // set default stream for dumping and tracing
    void setDumpStream(std::ostream* ostrm) const;
    std::ostream* getDumpStream() const;
    std::ostream& dump(std::ostream* strm=nullptr) const;
    std::ostream& dumpArg(std::ostream& strm, Arg arg, int indent, bool comma=true, bool dump_details=false) const;
    int indent() const;

    ModelFormat modelFormat() const;
    OnnxInfo getOnnxInfo() const;
    void setOnnxInfo(const OnnxInfo& info);

    struct Impl;
    Impl* impl() const;
private:
    std::shared_ptr<Impl> p;
};

struct CV_EXPORTS OnnxReaderParams
{
    int defaultWeightType = -1;
    std::string backendspec = std::string();
    std::ostream* errstrm = nullptr;
};

CV_EXPORTS Net2 readNetFromONNX2(std::string_view filename, const OnnxReaderParams& params={});

}}

#include "opencv2/dnn/op.hpp"

#endif
