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

struct InferTypes
{
    InferTypes(Net2* net_) : net(net_), netimpl(net_->impl()) {}

    void infer()
    {
        // reset types first
        size_t i, nargs = netimpl->args.size();
        for (i = 0; i < nargs; i++) {
            ArgInfo& info = netimpl->args[i];
            if (info.kind != DNN_ARG_INPUT && info.kind != DNN_ARG_CONST) {
                info.type = -1;
            }
        }
        inferGraph(netimpl->mainGraph);
    }

    bool inferGraph(const Graph& graph)
    {
        const std::vector<Node>& prog = graph->prog();
        size_t i, j, nargs = netimpl->args.size(), nops = prog.size();
        std::vector<int> inptypes, outtypes;

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
            inptypes.resize(ninputs);
            outtypes.resize(noutputs);

            for (j = 0; j < ninputs; j++) {
                const ArgInfo& info = netimpl->args[inputs[j].idx];
                inptypes[j] = info.type;
            }
            op->inferTypes(*net, graph, inputs, inptypes, outputs, outtypes);
            for (j = 0; j < noutputs; j++) {
                ArgInfo& info = netimpl->args[outputs[j].idx];
                info.type = outtypes[j];
            }
        }
    }

    Net2* net;
    Net2::Impl* netimpl;
};

void Net2::Impl::inferTypes()
{
    InferTypes typeInferencer(net);
    typeInferencer.infer();
}

}}
