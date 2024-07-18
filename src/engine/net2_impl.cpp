// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/net2_impl.hpp"

namespace cv { namespace dnn {

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
    prepared = false;

    strm = &std::cout;
    dump_indent = 3;

    clear();
}

Net2::Impl::~Impl() { clear(); }

void Net2::Impl::prepareForInference()
{
    if (!prepared) {
        constFold();
        inferTypes();
        constArgs();
        inferShapes(true);
        fuse();
        useBlockLayout();
        inferShapes(true);
        assignBuffers();
        prepared = true;
    }
}

void Net2::Impl::clear()
{
    modelFormat = DNN_MODEL_GENERIC;

    dimnames = NamesHash();
    dimnames_ = std::vector<std::string>();

    args = std::vector<ArgInfo>();
    argnames = NamesHash();

    tensors = std::vector<Tensor>();
    bufidxs = std::vector<int>();
    buffers = std::vector<Buffer>();

    mainGraph = Graph();

    pattern_args = std::vector<ArgInfo>();
    pattern_tensors = std::vector<Tensor>();

    ArgInfo info;
    args.push_back(info);
    pattern_args.push_back(info);
    tensors.push_back(Tensor());
    bufidxs.push_back(-1);
}

void Net2::Impl::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs)
{
    if (!mainGraph) {
        CV_Error(Error::StsNullPtr, "the model was not loaded");
    }
    // [TODO] initialize profile, tracer, symbolic shapes etc.
    size_t nsymdims = dimnames_.size();
    dimvalues.assign(nsymdims, -1);
    forwardGraph(mainGraph, inputs, outputs);
}

void Net2::Impl::checkAndUpdateDim(const Graph& g, const Node& node, Arg inp, int j, int64_t value)
{
    const ArgInfo& info = args[inp.idx];
    int64_t value0 = info.size.size[j];
    if (value0 >= 0) {
        if (value0 != value) {
            CV_Error_(Error::StsBadArg, ("graph '%s': node '%s': %d-th dimension of argument '%s' is wrong: %lld given, %lld expected",
                                        g->name().data(), node ? node->name().data() : "none (graph input)", j, info.name.c_str(), value, value0));
        }
    } else {
        int64_t idx = -value0-1;
        CV_Assert(0 <= idx && idx < (int64_t)dimvalues.size());
        value0 = dimvalues[idx];
        if (value0 < 0) {
            dimvalues[idx] = value;
        } else if (value0 != value) {
            CV_Error_(Error::StsBadArg,
            ("graph '%s': node '%s': %d-th dimension '%s' of argument '%s' is wrong: %lld given, but '%s' is already set to %lld",
                    g->name().data(), node ? node->name().data() : "none (graph input)",
                    j, dimnames_[idx].c_str(), info.name.c_str(),
                    value, dimnames_[idx].c_str(), value0));
        }
    }
}

void Net2::Impl::forwardGraph(Graph& g, const std::vector<Tensor>& inputs_,
                              std::vector<Tensor>& outputs_)
{
    const std::vector<Node>& prog = g->prog();
    size_t i, j, nops = prog.size();
    const std::vector<Arg>& gr_inputs = g->inputs();
    const std::vector<Arg>& gr_outputs = g->outputs();
    size_t n_gr_inputs = gr_inputs.size(), n_gr_outputs = gr_outputs.size();
    std::vector<Tensor> t_inputs, t_outputs;

    if (inputs_.size() != n_gr_inputs) {
        CV_Error_(Error::StsBadArg, ("wrong number of inputs in graph '%s': %zu given, %zu expected",
                                     g->name().data(), inputs_.size(), n_gr_inputs));
    }

    for (i = 0; i < n_gr_inputs; i++) {
        // [TODO] add conversion if needed
        const Tensor& t = inputs_[i];
        int ttype = t.type();
        TensorSize tsize = t.size();
        Arg inp = gr_inputs[i];
        const ArgInfo& info = args[inp.idx];
        if (info.type != ttype) {
            CV_Error_(Error::StsBadArg, ("wrong type of argument '%s': %s given, %s expected",
                                         info.name.c_str(), typeToString(ttype).c_str(),
                                         typeToString(info.type).c_str()));
        }

        if (info.size.ndims != tsize.ndims) {
            CV_Error_(Error::StsBadArg, ("wrong dimensionality of argument '%s': %d given, %d expected",
                                         info.name.c_str(), tsize.ndims, info.size.ndims));
        }
        
        for (int k = 0; k < tsize.ndims; k++) {
            checkAndUpdateDim(g, Node(), inp, k, tsize.size[k]);
        }

        if (info.kind == DNN_ARG_INPUT) {
            tensors[inp.idx] = t;
        } else if (info.kind == DNN_ARG_TEMP) {
            int bufidx = bufidxs[inp.idx];
            Tensor temp;
            temp.setBuffer(buffers[bufidx]);
            t.copyTo(temp);
            tensors[inp.idx] = temp;
        } else {
            CV_Error_(Error::StsBadArg, ("graph %s: argument '%s' must be 'INPUT' or 'TEMP', not '%s'",
                                         g->name().data(), info.name.c_str(), argKindToString(info.kind).c_str()));
        }
    }

    for (j = 0; j < nops; j++) {
        const Node& node = prog[j];
        if (!node)
            continue;
        printf("running '%s' (%s)\n", node->name().data(), node->op()->name().data());
        const std::vector<Arg>& inputs = node->inputs();
        const std::vector<Arg>& outputs = node->outputs();
        size_t ninputs = inputs.size(), noutputs = outputs.size();

        t_inputs.resize(ninputs);
        for (i = 0; i < ninputs; i++) {
            Arg inp = inputs[i];
            //const ArgInfo& info = args[inp.idx];
            t_inputs[i] = tensors[inp.idx];
        }

        t_outputs.resize(noutputs);
        for (i = 0; i < noutputs; i++) {
            Arg out = outputs[i];
            const ArgInfo& info = args[out.idx];
            if (info.kind == DNN_ARG_TEMP) {
                int bufidx = bufidxs[out.idx];
                t_outputs[i] = Tensor(buffers[bufidx]);
            } else {
                t_outputs[i] = tensors[out.idx];
            }
        }

        // [TODO] handle If/Loop/...
        CV_Assert(node->subgraphs().empty());
        Op& op = node->op();
        op->forward(*net, g, t_inputs, t_outputs, scratch_bufs);
        CV_Assert(t_outputs.size() == noutputs);

        for (i = 0; i < noutputs; i++) {
            Arg out = outputs[i];
            ArgInfo& info = args[out.idx];
            const Tensor& t = t_outputs[i];
            info.type = t.type();
            info.size = t.size();
            tensors[out.idx] = t;
            if (info.kind == DNN_ARG_TEMP) {
                int bufidx = bufidxs[out.idx];
                buffers[bufidx] = t.buffer();
            }
        }
    }

    outputs_.resize(n_gr_outputs);
    for (i = 0; i < n_gr_outputs; i++) {
        Arg out = gr_outputs[i];
        outputs_[i] = tensors[out.idx];
    }
}

void Net2::Impl::updateUseCounts(const Graph& graph, std::vector<int>& usecounts) const
{
    if (!graph)
        return;
    const std::vector<Node>& prog = graph->prog();
    for (const Node& node: prog) {
        const std::vector<Arg>& inputs = node->inputs();
        for (const Arg& input: inputs) {
            CV_Assert(input.idx < (int)usecounts.size());
            usecounts[input.idx]++;
        }
        const std::vector<Graph>& subgraphs = node->subgraphs();
        for (const Graph& subgraph: subgraphs) {
            updateUseCounts(subgraph, usecounts);
        }
    }
}

void Net2::Impl::useCounts(std::vector<int>& usecounts) const
{
    size_t nargs = args.size();
    usecounts.assign(nargs, 0);
    usecounts[0] = 1; // empty Arg() is always useful
    updateUseCounts(mainGraph, usecounts);
}

void Net2::Impl::checkArgs(const std::vector<Arg>& args_) const
{
    for (const Arg& a: args_) {
        checkArg(a);
    }
}

void Net2::Impl::checkArg(Arg a) const
{
    CV_Assert(a.idx >= 0);
    CV_Assert(a.idx < (int)args.size());
}

}}
