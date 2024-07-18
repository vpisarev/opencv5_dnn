// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/net2_impl.hpp"

namespace cv { namespace dnn {

using std::vector;
using std::string;

typedef std::pair<int, int> int_pair;
typedef std::pair<int, Arg> int_arg_pair;

struct InferShapes
{
    InferShapes(Net2* net_, bool symbolic_) : net(net_), netimpl(net_->impl()), symbolic(symbolic_) {}

    void infer()
    {
        inferGraph(netimpl->mainGraph);
    }

    bool inferGraph(const Graph& graph)
    {
        const std::vector<Node>& prog = graph->prog();
        size_t i, j, nargs = netimpl->args.size(), nops = prog.size();
        std::vector<TensorSize> inpshapes, outshapes;

        for (i = 0; i < nops; i++) {
            const Node& node = prog[i];
            std::vector<Graph>& subgraphs = const_cast<std::vector<Graph>&>(node->subgraphs());
            for (Graph& g: subgraphs) {
                inferGraph(g);
            }
            const std::vector<Arg>& inputs = node->inputs();
            const std::vector<Arg>& outputs = node->outputs();
            size_t ninputs = inputs.size();
            size_t noutputs = outputs.size();
            const Op& op = node->op();
            inpshapes.resize(ninputs);
            outshapes.resize(noutputs);

            bool haveSymbols = false;
            for (j = 0; j < ninputs; j++) {
                const ArgInfo& info = netimpl->args[inputs[j].idx];
                inpshapes[j] = info.size;
                haveSymbols = haveSymbols || info.size.haveSymbols();
            }
            if (symbolic)
                CV_Assert(!haveSymbols);
            op->inferShapes(*net, graph, inputs, inpshapes, outputs, outshapes, haveSymbols);
            for (j = 0; j < noutputs; j++) {
                ArgInfo& info = netimpl->args[outputs[j].idx];
                info.size = outshapes[j];
            }
        }
    }

    Net2* net;
    Net2::Impl* netimpl;
    bool symbolic;
};

void Net2::Impl::inferShapes(bool symbolic)
{
    InferShapes shapeInferencer(net, symbolic);
    shapeInferencer.infer();
}

}}
