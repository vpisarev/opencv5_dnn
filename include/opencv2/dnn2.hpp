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

enum DataLayout2
{
    DNN2_LAYOUT_UNKNOWN = 0,
    DNN2_LAYOUT_ND = 1,
    DNN2_LAYOUT_NCHW = 2,
    DNN2_LAYOUT_NHWC = 3,
    DNN2_LAYOUT_NCHWc = 4
};

struct CV_EXPORTS TensorShape
{
    TensorShape();
    template<typename _Tp> TensorShape(int ndims, const _Tp* shape,
                                        DataLayout2 layout=DNN2_LAYOUT_UNKNOWN);
    template<typename _Tp> TensorShape(std::initializer_list<_Tp> shape,
                                        DataLayout2 layout=DNN2_LAYOUT_UNKNOWN);
    static TensorShape fromArray(InputArray m, DataLayout2 layout=DNN2_LAYOUT_UNKNOWN);
    int toMatShape(int* mshape, int maxdims) const;
    size_t total() const;
    bool empty() const;
    bool equalTo(const TensorShape& another) const;
    std::string str() const;
    void dump(std::ostream& strm) const;
    enum {MAX_TENSOR_DIMS=10};
    DataLayout2 layout;
    int ndims;
    int64_t C;
    int64_t shape[MAX_TENSOR_DIMS];
};

struct CV_EXPORTS ShapeNType
{
    TensorShape shape;
    int type;
    bool totalBytes() const;
    ShapeNType toBlock(int64_t C0) const;
    ShapeNType fromBlock(int64_t C) const;

    void dump(std::ostream& strm) const;
};

CV_EXPORTS bool operator == (const ShapeNType& st0, const ShapeNType& st1);
CV_EXPORTS bool operator != (const ShapeNType& st0, const ShapeNType& st1);

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
    virtual ~Device();
    virtual DeviceType type() const = 0;
    virtual std::string name() const = 0;
    virtual bool supportType(int type) const = 0;
    virtual bool zeroCopy() const = 0;
    virtual int ndevices() const = 0;
    virtual int index() const = 0;
    virtual Device* sameKindDevice(int index) const = 0;
    virtual MemoryManager* defaultMemoryManager() = 0;
    virtual bool sameDevice(const Device* device) const = 0;
};

CV_EXPORTS Device* getCPUDevice();

struct CV_EXPORTS MemoryManager
{
    virtual ~MemoryManager();
    virtual void* allocate(Device* device, size_t bufsize) = 0;
    virtual void release(Device* device, void* handle) = 0;
    virtual void* map(Device* device, void* handle, size_t size, int access=DNN_BUF_RW) = 0;
    virtual void unmap(Device* device, void* handle, void* ptr, size_t size, int access=DNN_BUF_RW) = 0;
    virtual void copyFromDevice(Device* device, void* handle, size_t offset, size_t size, void* dst) = 0;
    virtual void copyToDevice(Device* device, const void* src, void* handle, size_t offset, size_t size) = 0;
    virtual void fill(Device* device, void* handle, size_t offset, size_t size, const void* value, size_t vsize) = 0;
};

struct CV_EXPORTS Buffer
{
    static constexpr size_t whole = (size_t)-1;
    struct Shared
    {
        Shared();
        void* ptr;
        int refcount;
        int mapcount;
    };
    Buffer();
    Buffer(const Buffer& buf);
    Buffer(const void* data, size_t size, bool copy);
    Buffer& operator = (const Buffer& buf);
    ~Buffer();

    static Buffer allocate(size_t size, Device* device=nullptr, MemoryManager* mm=nullptr);
    void fit(size_t size);
    void set(const void* data, size_t size, bool copy);
    void release();
    void* map(BufAccess access=DNN_BUF_RW);
    void unmap(BufAccess access=DNN_BUF_RW);

    Device* device;
    MemoryManager* mm;
    Shared* shared;
    void* handle;
    size_t size;
};

// temporary solution while Mat cannot be 0-D or 1-D array.
struct CV_EXPORTS Tensor
{
    Tensor();
    Tensor(const Tensor& t);
    Tensor& operator = (const Tensor& t);
    ~Tensor();

    explicit Tensor(const ShapeNType& st, Device* device=nullptr);
    explicit Tensor(const ShapeNType& st, void* data, bool copy, Device* device=nullptr);
    explicit Tensor(const ShapeNType& st, Buffer& buffer, size_t slice_start=0);
    explicit Tensor(const ShapeNType& st, void* data, Buffer& buffer, size_t slice_start=0);
    explicit Tensor(InputArray arr, bool copy, Device* device=nullptr);
    explicit Tensor(InputArray arr, Buffer& buffer, size_t slice_start=0);
    explicit Tensor(Buffer& buffer, size_t slice_size=Buffer::whole, size_t slice_start=0);

    static void multiFit(Buffer& buffer,
                        std::initializer_list<ShapeNType> st,
                        std::initializer_list<Tensor*> tensors);
    static Tensor makeScalar(double value, int type, Device* device=nullptr);
    static Tensor makeScalar(int64 value, int type, Device* device=nullptr);
    bool isScalar() const;
    bool getScalar(int type, void* scalar) const;
    template<typename _Tp> _Tp getScalar() const {
        _Tp value = 0;
        bool ok = getScalar(DataType<_Tp>::depth, &value);
        return ok ? value : _Tp();
    }

    void release();
    void fit(const ShapeNType& st);
    void fit(const ShapeNType& st, Buffer& buffer, size_t slice_start=0);
    void fitSameDevice(const Tensor& tensor, const ShapeNType& shapetyp);
    bool onSameDevice(const Tensor& tensor);
    void set(const ShapeNType& st, void* data, bool copy);
    void set(InputArray arr, bool copy);
    void set(Buffer& buffer, size_t slice_start=0);

    Device* device() const;
    DeviceType deviceType() const;
    size_t total() const;
    size_t elementSize() const;
    bool empty() const;
    void* data() const;
    Mat getMat() const;
    Mat download() const;
    Tensor upload(Device* device) const;
    Tensor uploadToSameDevice(const Tensor& t) const;
    void copyTo(Tensor& tensor) const;
    void setTo(double scalar);
    void* map(int access=DNN_BUF_RW);
    void unmap(int access=DNN_BUF_RW);

    void dump(std::ostream& strm, int indent, int context=3,
              int maxsz_all=100, bool braces=true) const;
    void dumpSmall(std::ostream& strm, int maxsz_small=10, bool braces=true) const;

    int flags;
    ShapeNType st;
    Buffer buf;
    size_t slice_size;
    size_t slice_start;
};

class Net2;
struct Node;
struct Graph;
typedef Ptr<Graph> PGraph;
typedef Ptr<Node> PNode;
struct BaseBackend;
typedef Ptr<BaseBackend> PBackend;

struct CV_EXPORTS Arg
{
    Arg();
    explicit Arg(int idx);
    bool isPattern() const;
    int idx;
};

struct CV_EXPORTS BaseOp
{
public:
    virtual ~BaseOp();
    virtual std::string_view origname() const;
    virtual std::string_view name() const;
    virtual void dump(std::ostream& strm, int indent, int maxsz_small=10) const;
    static void dumpTensorAttr(std::ostream& strm, std::string_view name, const Tensor& t, int indent);
    static void dumpScalarAttr_(std::ostream& strm, std::string_view name, int type, const void* scalar, int indent);
    template<typename _Tp> static void dumpScalarAttr(std::ostream& strm, std::string_view name, _Tp scalar, int indent)
    { dumpScalarAttr_(strm, name, DataType<_Tp>::depth, &scalar, indent); }
    static void dumpStringAttr(std::ostream& strm, std::string_view name, std::string_view value, int indent);

    virtual void clone() const;
    virtual int minNumInputs() const;
    virtual int maxNumInputs() const;
    virtual int minNumOutputs() const;
    virtual int maxNumOutputs() const;

    virtual bool supportType(int input, int depth) const;
    virtual bool supportInplace(const Net2& net, const Graph& graph,
                                const std::vector<Arg>& inpargs,
                                const std::vector<ShapeNType>& inpst) const;

    virtual int64 getFLOPS(const std::vector<ShapeNType> &inputs,
                           const std::vector<ShapeNType> &outputs) const;
    virtual void inferShapes(const Net2& net, const Graph& graph,
                            const std::vector<Arg>& inpargs,
                            const std::vector<ShapeNType>& inpst,
                            const std::vector<Arg>& outargs,
                            std::vector<ShapeNType>& outst,
                            std::vector<size_t>& tempbufs) const;
    virtual void forward(Net2& net, Graph& graph,
                        const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs,
                        std::vector<Buffer>& tempbufs);
};

typedef Ptr<BaseOp> Op;

struct CV_EXPORTS Node
{
    void dump(const Net2& net, std::ostream& strm,
              int indent, int maxsz_small=10) const;

    std::string name;
    Op op;
    std::vector<Arg> inputs;
    std::vector<Arg> outputs;
    std::vector<PGraph> graph;
};

struct CV_EXPORTS OptimizedGraph
{
    virtual ~OptimizedGraph();
    virtual Backend* getBackend() const = 0;
    virtual void dump(const Net2& net, const PGraph& g,
                      std::ostream& strm, int indent, int maxsz_small=10) const;
    virtual bool update(Net2& net, const PGraph& g,
                        const std::vector<ShapeNType>& curr_inpst,
                        const std::vector<ShapeNType>& prev_inpst,
                        std::vector<Buffer>& tempbufs) const;
    virtual bool forward(Net2& net, const PGraph& graph, std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs, std::vector<Buffer>& tempbufs) = 0;
};

typedef Ptr<OptimizedGraph> POptiGraph;

struct CV_EXPORTS Graph
{
    Graph();
    Graph(Net2& net_, std::string_view name_, bool ispattern_=false);
    bool empty() const;
    void clear();
    PGraph clone(Net2* newnet=nullptr) const;
    void newop(const Op& op, const std::vector<Arg>& inputs, std::vector<Arg>& outputs);
    Arg newop(const Op& op, const std::vector<Arg>& inputs);
    bool isPattern() const;
    void dump(std::ostream& strm, int indent);
    void inferShapes(const std::vector<ShapeNType>& inpst,
                    std::vector<ShapeNType>& outst) const;

    std::vector<Arg> inputs;
    std::vector<Arg> outputs;
    std::vector<PNode> prog;

    std::vector<ShapeNType> prev_inpst;

    Net2* net;
    PBackend backend;
    POptiGraph optimized;

    std::string name;
    bool ispattern;
};

struct CV_EXPORTS BaseBackend
{
    virtual ~BaseBackend();
    virtual Device* device() const = 0;
    virtual std::string_view name() const = 0;
    virtual bool supportType(int type) const = 0;
    virtual int64_t preferredBlockSize(int type) const = 0;
    virtual bool supportOp(const Op& op, const std::vector<ShapeNType>& inpst) const = 0;
    virtual PGraph preprocessGraph(Net2& net, const PGraph& graph,
                        const std::vector<ShapeNType>& inpst,
                        std::vector<Buffer>& tempbufs) = 0;
    virtual bool forward(Net2& net, const PGraph& graph, std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs, std::vector<Buffer>& tempbufs) = 0;
};

CV_EXPORTS PBackend getCPUBackend();
CV_EXPORTS PBackend getBackendFromSpec(std::string_view backendSpec);

enum ArgKind { DNN_ARG_CONST=0, DNN_ARG_INPUT=1, DNN_ARG_OUTPUT=2, DNN_ARG_TEMP=3 };

struct CV_EXPORTS ArgInfo
{
    ArgInfo();
    std::string name;
    ArgKind kind;
    ShapeNType st;
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

class CV_EXPORTS_W_SIMPLE Net2
{
public:
    Net2();  //!< Default constructor.
    ~Net2(); //!< Destructor frees the net only if there aren't references to the net anymore.

    bool preprocessWith(std::initializer_list<Op> ops);
    bool postprocessWith(std::initializer_list<Op> ops);
    void forward(InputArrayOfArrays inputBlobs,
                 OutputArrayOfArrays outputBlobs);
    void getInputNames(std::vector<std::string>& inputs) const;
    void getOutputNames(std::vector<std::string>& outputs) const;

    void setTracingMode(TracingMode mode);
    TracingMode getTracingMode() const;
    void setProfilingMode(ProfilingMode mode);
    void getProfile(std::vector<std::string>& opnames, std::vector<double>& times) const;
    ProfilingMode getProfilingMode() const;
    PGraph newGraph(std::string_view name=std::string_view()) const;
    PGraph newPattern() const;
    PGraph getMainGraph() const;
    bool setAccuracy(int type); // typically, CV_16F can be used to explicitly
                                // enable FP16 on backends that support both FP16 and FP32
    int getAccuracy() const;
    void checkArgs(const std::vector<Arg>& args) const;
    void checkArg(Arg arg) const;
    ArgInfo argInfo(Arg arg) const;
    Arg newArg(const ArgInfo& info) const;
    Arg newConstArg(std::string_view name, const Tensor& t) const;
    Arg newTempArg(std::string_view name=std::string_view()) const;
    bool isConstArg(Arg idx) const;
    bool isTempArg(Arg idx) const;
    Tensor argTensor(Arg idx) const;

    int onnxOpset() const;

    bool useBackend(std::string_view backendSpec); // CUDA:1, iGPU, NPU:0, ...
    bool useBackend(const PBackend& backend); // the latest added backend gets the highest priority
    void getBackends(std::vector<PBackend>& backends); // highest-priority to the lowest-priority
    size_t getNumUsedBackends() const;
    PBackend getBackend(size_t i) const; // 0 - highest-priority backend, ...

    Net2 clone() const;
    bool empty() const;

    void setDumpStream(std::ostream* ostrm) const;
    std::ostream* getDumpStream() const;
    void dump(std::ostream* strm=nullptr) const;
    void dumpArg(std::ostream& strm, Arg arg, int maxsz_small=10) const;

    void setDumpIndent(int);
    int getDumpIndent() const;

    struct Impl;
    Ptr<Impl> impl() const;
private:
    Ptr<Impl> impl_;
};

struct CV_EXPORTS OnnxReaderParams
{
    int accuracy = -1;
    std::string backendspec = std::string();
    std::ostream* errstrm = nullptr;
};

CV_EXPORTS Net2 readNetFromONNX2(std::string_view filename, const OnnxReaderParams& params={});

}}

#include "opencv2/dnn/op.hpp"

#endif
