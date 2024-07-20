// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/net2_impl.hpp"

namespace cv { namespace dnn {

ArgInfo::ArgInfo()
{
    kind = DNN_ARG_EMPTY;
    type = -1;
}

Net2::Net2()
{
    p = std::make_shared<Impl>(this);
}

Net2::~Net2()
{
    p = std::shared_ptr<Impl>();
}

Net2::Impl* Net2::impl() const { return p.get(); }

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
    int i, ninputs = (int)inputBlobs.total();
    std::vector<Mat> inp_ms(ninputs), out_m;
    std::vector<Tensor> inputs(ninputs), outputs;
    for (i = 0; i < ninputs; i++) {
        inp_ms[i] = inputBlobs.getMat(i);
        int ndims = inp_ms[i].dims;
        TensorLayout layout = ndims >= 3 ? p->defaultLayout : ndims == 2 ? LAYOUT_ND : LAYOUT_UNKNOWN;
        inputs[i] = Tensor(inp_ms[i], layout, false, nullptr); // inp_ms exist until forward() is finished,
                                                 // so we have input tensors protected from premature release.
                                                 // Therefore, we don't need to create extra copy of those tensors,
                                                 // we just create temporary Tensor headers on top of this data.
    }
    p->forward(inputs, outputs);
    int noutputs = (int)outputs.size();
    // [TODO] eliminate output data copy: many of computer vision models produce much smaller input than output,
    // but some, like superresolution, may have output bigger than input, so it would be useful to eliminate it.
    std::vector<Mat>& out_ms = outputBlobs.getMatVecRef();
    out_ms.resize(noutputs);
    for (i = 0; i < noutputs; i++) {
        outputs[i].getMat().copyTo(out_ms[i]);
    }
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

Arg Net2::getArg(std::string_view name)
{
    if (!name.empty()) {
        std::string name_(name);
        auto it = p->argnames.find(name_);
        if (it != p->argnames.end()) {
            return Arg((int)it->second);
        }
    }
    return newArg(name, DNN_ARG_TEMP);
}

bool Net2::haveArg(std::string_view name) const
{
    std::string name_(name);
    return p->argnames.find(name_) != p->argnames.end();
}

Arg Net2::newConstArg(std::string_view name, const Tensor& t) const
{
    Arg arg = newArg(name, DNN_ARG_CONST);
    p->tensors[arg.idx] = t;
    ArgInfo& info = p->args[arg.idx];
    info.type = t.type();
    info.size = t.size();
    return arg;
}

Arg Net2::newArg(std::string_view name, ArgKind kind) const
{
    int idx = (int)p->args.size();

    if (!name.empty()) {
        std::string name_(name);
        CV_Assert(p->argnames.find(name_) == p->argnames.end());
        p->argnames.insert(std::make_pair(name_, (int64_t)idx));
    }

    ArgInfo info;
    info.name = name;
    info.kind = kind;
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

int64_t Net2::findDim(std::string_view name, bool insert)
{
    std::string name_(name);

    if (!name.empty()) {
        auto it = p->dimnames.find(name_);
        if (it != p->dimnames.end())
            return it->second;
    }
    if (!insert) {
        CV_Error(Error::StsObjectNotFound, "");
    }

    int64_t dim = -(int64_t)p->dimnames_.size()-1;
    int attempt = 0, max_attempt = 100000;
    for (;attempt < max_attempt; attempt++) {
        if (name.empty())
            name_ = attempt == 0 ? format("N%d", (int)dim) : format("N%d.%d", (int)dim, attempt);
        auto it = p->dimnames.find(name_);
        if (it != p->dimnames.end()) {
            p->dimnames.insert(std::make_pair(name_, dim));
            p->dimnames_.push_back(name_);
            break;
        }
    }
    CV_Assert(attempt < max_attempt);
    return dim;
}


std::string Net2::dimToString(int64_t dim) const
{
    if (dim >= 0)
        return format("%lld", (long long)dim);
    int64_t idx = -dim-1;
    CV_Assert(idx < (int64_t)p->dimnames_.size());
    return p->dimnames_[idx];
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

std::ostream& Net2::dumpArg(std::ostream& strm, Arg arg, int indent, bool comma, bool dump_details) const
{
    const ArgInfo& info = argInfo(arg);
    prindent(strm, indent);
    if (arg.empty()) {
        strm << "<empty>" << (comma ? "," : "");
    } else {
        strm << '\"' << info.name << (comma ? "\"," : "\"");
        if (dump_details && arg.idx > 0) {
            strm << " // ";
            strm << (info.kind == DNN_ARG_INPUT ? "<Input>" :
                     info.kind == DNN_ARG_OUTPUT ? "<Output>" :
                     info.kind == DNN_ARG_CONST ? "<Const>" :
                     info.kind == DNN_ARG_TEMP ? "<Temp>" :
                     "<Uknown kind ???>");
            if (info.type >= 0) {
                strm << " " << typeToString(info.type);
                if (info.size.empty()) {
                    strm << " <empty>";
                } else {
                    if (info.size.ndims > 0 && info.size.layout != LAYOUT_UNKNOWN) {
                        strm << " " << layoutToString(info.size.layout);
                    }
                    strm << " [";
                    for (int i = 0; i < info.size.ndims; i++) {
                        strm << (i > 0 ? " x " : "");
                        strm << dimToString(info.size.size[i]);
                    }
                    strm << "]";
                }
            }
            if (info.kind == DNN_ARG_TEMP)
                strm << " (buf #" << p->bufidxs[arg.idx] << ")";
        }
    }
    strm << "\n";
    return strm;
}

int Net2::indent() const { return p->dump_indent; }
ModelFormat Net2::modelFormat() const { return p->modelFormat; }

OnnxInfo Net2::getOnnxInfo() const { return p->onnxInfo; }
void Net2::setOnnxInfo(const OnnxInfo& info) {
    p->modelFormat = DNN_MODEL_ONNX;
    p->defaultLayout = LAYOUT_NCHW;
    p->onnxInfo = info;
}

void Net2::prepare()
{
    p->prepareForInference();
}

}}
