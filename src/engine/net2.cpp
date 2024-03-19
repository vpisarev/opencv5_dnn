// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"

namespace cv { namespace dnn {

ArgInfo::ArgInfo()
{
    kind = DNN_ARG_EMPTY;
    type = 0;
}

void Net2::Impl::initArgs()
{
    ArgInfo info;
    args.push_back(info);
    pattern_args.push_back(info);
    tensors.push_back(Tensor());
    bufidxs.push_back(-1);
}

Net2::Impl::Impl(Net2* net_)
{
    CV_Assert(net_ != nullptr);
    net = net_;
    modelFormat = DNN_MODEL_GENERIC;
    defaultLayout = LAYOUT_NCHW;
    onnx_opset = 0;

    defaultDevice = Device::CPU();
    defaultMemoryManager = MemoryManager::forCPU();

    accuracy = CV_32F;
    enableFP16 = haveFP16 = false;
    if (checkHardwareSupport(CV_CPU_FP16)) {
        enableFP16 = haveFP16 = true;
    }

    tracingMode = DNN_TRACE_NONE;
    profilingMode = DNN_PROFILE_NONE;

    strm = &std::cout;
    dump_indent = 3;
}

Net2::Impl::~Impl() { clear(); }

void Net2::Impl::clear()
{
    modelFormat = DNN_MODEL_GENERIC;

    argnames = NamesHash();
    dimnames = NamesHash();
    dimnames_ = std::vector<std::string>();
    args = std::vector<ArgInfo>();
    tensors = std::vector<Tensor>();
    bufidxs = std::vector<int>();
    buffers = std::vector<Buffer>();
    mainGraph = Graph();
}

void Net2::Impl::forwardGraph(const Graph&)
{
    CV_Error(Error::StsNotImplemented, "");
}

void Net2::Impl::useCounts(std::vector<int>&) const
{
    CV_Error(Error::StsNotImplemented, "");
}

void Net2::Impl::updateUseCounts(std::vector<int>&, const Graph&) const
{
    CV_Error(Error::StsNotImplemented, "");
}

void Net2::Impl::checkArgs(const std::vector<Arg>&) const
{
    CV_Error(Error::StsNotImplemented, "");
}

void Net2::Impl::checkArg(Arg) const
{
    CV_Error(Error::StsNotImplemented, "");
}

void Net2::Impl::assignBuffers()
{
    CV_Error(Error::StsNotImplemented, "");
}

void Net2::Impl::fuse()
{
    CV_Error(Error::StsNotImplemented, "");
}

void Net2::Impl::foldConstSubexpr()
{
    CV_Error(Error::StsNotImplemented, "");
}

Net2::Net2()
{
    p = std::make_shared<Impl>(this);
}

Net2::~Net2()
{
    p = std::shared_ptr<Impl>();
}

void Net2::release()
{
    // create complete copy of the internal representation, but the clear one.
    // in this case 'Net2 n2 = n1; ... n2.release();' will not affect n1.
    std::shared_ptr<Impl> p2 = std::make_shared<Impl>(this);
    *p2 = *p;
    p2->clear();
    p = p2;
}

void Net2::forward(InputArrayOfArrays inputBlobs,
                   OutputArrayOfArrays outputBlobs)
{
    CV_Error(Error::StsNotImplemented, "");
}

void Net2::getInputNames(std::vector<std::string>& inputs) const
{
    auto inputs_ = getMainGraph()->inputs();
    inputs.clear();
    for (auto inp: inputs_) {
        inputs.push_back(argInfo(inp).name);
    }
}

void Net2::getOutputNames(std::vector<std::string>& outputs) const
{
    auto outputs_ = getMainGraph()->outputs();
    outputs.clear();
    for (auto out: outputs_) {
        outputs.push_back(argInfo(out).name);
    }
}

void Net2::setTracingMode(TracingMode mode)
{
    p->tracingMode = mode;

}

TracingMode Net2::getTracingMode() const
{
    return p->tracingMode;
}

void Net2::setProfilingMode(ProfilingMode mode)
{
    p->profilingMode = mode;
}

ProfilingMode Net2::getProfilingMode() const
{
    return p->profilingMode;
}

void Net2::getProfile(std::vector<std::string>& opnames,
                      std::vector<double>& times) const
{
    size_t nentries = p->profileNames.size();
    opnames.resize(nentries);
    times.resize(nentries);

    for (size_t i = 0; i < nentries; i++) {
        opnames[i] = p->profileNames[i];
        times[i] = p->profileTimes[i];
    }
}

Graph Net2::newGraph(std::string_view name,
                     const std::vector<std::string>& inpnames,
                     const std::vector<std::string>& outnames) const
{
    std::vector<Arg> inputs, outputs;
    bool maingraph = !p->mainGraph;
    for (std::string_view inpname : inpnames) {
        CV_Assert(!inpname.empty());
        inputs.push_back(newArg(inpname, (maingraph ? DNN_ARG_INPUT : DNN_ARG_TEMP)));
    }
    for (std::string_view outname : outnames) {
        CV_Assert(!outname.empty());
        outputs.push_back(newArg(outname, (maingraph ? DNN_ARG_OUTPUT : DNN_ARG_TEMP)));
    }
    Graph g = std::make_shared<GraphData>(*this, name, inputs, false);
    g->setOutputs(outputs);
    if (maingraph)
        p->mainGraph = g;

    return g;
}

Graph Net2::newGraph(std::string_view name,
                     const std::vector<Arg>& inputs) const
{
    bool maingraph = !p->mainGraph;
    for (Arg inp : inputs) {
        ArgKind inp_kind = argKind(inp);
        CV_Assert(inp_kind == (maingraph ? DNN_ARG_INPUT : DNN_ARG_TEMP));
    }
    Graph g = std::make_shared<GraphData>(*this, name, inputs, false);
    if (maingraph)
        p->mainGraph = g;

    return g;
}

Graph Net2::newPatternGraph(std::string_view name,
                            const std::vector<Arg>& inputs) const
{
    for (Arg inp : inputs) {
        ArgKind inp_kind = argKind(inp);
        CV_Assert(inp_kind == DNN_ARG_PATTERN || inp_kind == DNN_ARG_CONST);
    }
    Graph g = std::make_shared<GraphData>(*this, name, inputs, true);

    return g;
}

Graph Net2::getMainGraph() const
{
    return p->mainGraph;
}

void Net2::setMainGraph(const Graph& g)
{
    CV_Assert(g && !g->isPattern());
    return p->mainGraph = g;
}

bool Net2::setAccuracy(int type)
{
    CV_Assert(type == CV_32F || type == CV_16F || type == CV_16BF);
    if ((type == CV_16F || type == CV_16BF) && p->haveFP16)
        p->accuracy = type;
    else
        p->accuracy = CV_32F;
}

int Net2::getAccuracy() const
{
    return p->accuracy;
}

void Net2::checkArgs(const std::vector<Arg>& args) const
{
    for (Arg arg: args) {
        argInfo(arg); // ignore the return value, just try to find information about particular argument.
    }
}

void Net2::checkArg(Arg arg) const
{
    argInfo(arg);
}

const ArgInfo& Net2::argInfo(Arg arg) const
{
    if (arg.idx >= 0) {
        CV_Assert(arg.idx < (int)p->args.size());
        return p->args[arg.idx];
    } else {
        CV_Assert(-arg.idx < (int)p->pattern_args.size());
        return p->pattern_args[-arg.idx];
    }
}

std::string_view Net2::argName(Arg arg) const { return argInfo(arg).name; }

ArgKind Net2::argKind(Arg arg) const { return argInfo(arg).kind; }

Arg Net2::getArg(std::string_view name) const
{
    if (!name.empty()) {
        std::string name_(name);
        auto it = p->argnames.find(name_);
        if (it != p->argnames.end()) {
            return Arg(it->second);
        }
    }
    return newArg(name, DNN_ARG_TEMP);
}

Arg Net2::newConstArg(std::string_view name, const Tensor& t) const
{
    Arg arg = newArg(name, DNN_ARG_CONST);
    p->tensors[arg.idx] = t;

    return arg;
}

Arg Net2::newArg(std::string_view name, ArgKind kind) const
{
    int idx = (int)p->args.size();

    if (!name.empty()) {
        std::string name_(name);
        CV_Assert(p->argnames.find(name_) == p->argnames.end());
        p->argnames.insert(std::make_pair(name_, idx));
    }

    ArgInfo info;
    info.name = name;
    info.kind = DNN_ARG_TEMP;
    p->args.push_back(info);
    p->tensors.push_back(Tensor());
    p->bufidxs.push_back(-1);

    return Arg(idx);
}

bool Net2::isConstArg(Arg arg) const
{
    return argKind(arg) == DNN_ARG_CONST;
}

bool Net2::isTempArg(Arg arg) const
{
    return argKind(arg) == DNN_ARG_TEMP;
}

bool Net2::isPattern(Arg arg) const
{
    return arg.idx < 0;
}

Tensor Net2::argTensor(Arg arg) const
{
    CV_Assert(0 <= arg.idx && arg.idx < (int)p->tensors.size());
    return p->tensors[arg.idx];
}

TensorSize Net2::argSize(Arg arg) const
{
    return argInfo(arg).size;
}

int Net2::argType(Arg arg) const
{
    return argInfo(arg).type;
}

SizeType Net2::argSizeType(Arg arg) const
{
    ArgInfo info = argInfo(arg);
    return SizeType({info.size, info.type});
}

bool Net2::useBackend(std::string_view backendSpec)
{
    GraphBackend* backend = GraphBackend::fromSpec(backendSpec);
    if (!backend)
        return false;
    return useBackend(backend);
}

bool Net2::useBackend(GraphBackend* backend)
{
    CV_Assert(backend != nullptr);
    size_t i = 0, j = 0, nbackends = p->backends.size();
    for (; i < nbackends; i++) {
        if (p->backends[i] != backend) {
            p->backends[j] = p->backends[i];
            j++;
        }
    }
    p->backends.push_back(backend);
    return true;
}

bool Net2::removeBackend(GraphBackend* backend)
{
    CV_Assert(backend != nullptr);
    size_t i = 0, j = 0, nbackends = p->backends.size();
    for (; i < nbackends; i++) {
        if (p->backends[i] != backend) {
            p->backends[j] = p->backends[i];
            j++;
        }
    }
    return true;
}

size_t Net2::getNumUsedBackends() const
{
    return p->backends.size();
}

GraphBackend* Net2::getBackend(size_t i) const
{
    size_t nbackends = p->backends.size();
    CV_Assert(i < nbackends);
    return p->backends[nbackends - i - 1];
}

Net2 Net2::clone() const
{
    CV_Error(Error::StsNotImplemented, "");
}

bool Net2::empty() const
{
    return !p->mainGraph || p->mainGraph->empty();
}

// set default stream for dumping and tracing
void Net2::setDumpStream(std::ostream* ostrm) const
{
    p->strm = ostrm ? ostrm : &std::cout;
}

std::ostream* Net2::getDumpStream() const
{
    return p->strm;
}

std::ostream& Net2::dump(std::ostream* strm0) const
{
    std::ostream& strm = strm0 ? *strm0 : *getDumpStream();
    Graph g = getMainGraph();
    if (!g) {
        strm << "{}\n";
    } else {
        g->dump(strm, 0, false);
    }
    return strm;
}

std::ostream& Net2::dumpArg(std::ostream& strm, Arg arg, int indent, bool comma) const
{
    return strm;
}

int Net2::indent() const { return p->dump_indent; }
ModelFormat Net2::modelFormat() const { return p->modelFormat; }
int Net2::onnxOpset() const { return p->onnx_opset; }

}}
