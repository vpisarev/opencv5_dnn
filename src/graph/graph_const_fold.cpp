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

struct ConstFolding
{
    ConstFolding(Net2* net_) : net(net_), netimpl(net_->impl()) {}

    void process()
    {
        size_t nargs = netimpl->args.size();
        netimpl->tensors.resize(nargs);
        netimpl->useCounts(usecounts);
        C0 = 8; // [TODO] netimpl->backends(0)
        processGraph(netimpl->mainGraph);
    }

    Node* getNode(std::vector<Node>& newprog, int op_idx) const
    {
        return op_idx >= 0 ? &newprog.at(op_idx) : 0;
    }

    template<typename _OpType> _OpType* getOp(const Node* n) const
    {
        return !n ? nullptr : dynamic_cast<_OpType*>((*n)->op().get());
    }

    void unuse(Arg inp)
    {
        CV_Assert(usecounts[inp.idx] > 0);
        if (--usecounts[inp.idx] == 0 && net->isConstArg(inp)) {
            netimpl->tensors[inp.idx] = Tensor(); // deallocate unused tensor
        }
    }

    bool processGraph(Graph& graph)
    {
        bool modified = false;
        const std::vector<Node>& prog = graph->prog();
        size_t i, nops = prog.size();
        std::vector<Node> newprog;
        std::vector<Buffer> temp;
        std::vector<Arg> removed_args;
        std::vector<Tensor> t_inputs;

        for (i = 0; i < nops; i++) {
            const Node& node = prog[i];
            std::vector<Graph>& subgraphs = const_cast<std::vector<Graph>&>(node->subgraphs());
            for (Graph& g: subgraphs) {
                if (processGraph(g))
                    modified = true;
            }
            const std::vector<Arg>& inputs = node->inputs();
            const std::vector<Arg>& outputs = node->outputs();
            const Op& op = node->op();
            size_t j, ninputs = inputs.size(), noutputs = outputs.size();
            bool all_const = true;
            t_inputs.assign(ninputs, Tensor());
            for (j = 0; j < ninputs; j++) {
                Arg inp = inputs[j];
                bool const_arg = net->isConstArg(inp);
                if (!const_arg)
                    all_const = false;
                if (all_const)
                    t_inputs[j] = netimpl->tensors.at(inp.idx);
            }

            if (all_const &&
                op->supportBlockLayout(0, (int)ninputs) <= 0 // we don't currently support constant folding
                                               // for block-layout operations (Convolution, MaxPool, AveragePool)
                ) {
                std::vector<Tensor> t_outputs(noutputs);
                op->forward(*net, graph, t_inputs, t_outputs, temp);
                CV_Assert(t_outputs.size() == noutputs);
                for (j = 0; j < noutputs; j++) {
                    Arg out = outputs[j];
                    ArgInfo& out_info = netimpl->args.at(out.idx);
                    out_info.type = t_outputs[j].type();
                    out_info.size = t_outputs[j].size();
                    out_info.kind = DNN_ARG_CONST; // re-classify each output as constant
                    netimpl->tensors.at(out.idx) = t_outputs[j];
                }

                modified = true;
                for (size_t i = 0; i < ninputs; i++)
                    unuse(inputs[i]);
                //printf("folded %s: %s\n", op->name().data(), node->name().data());
                // we don't add operation into the new program,
                // because the output of the all-const inputs operation is now a constant,
                // stored in a separate tensor
            } else {
                newprog.push_back(node);
            }
        }

        if (modified) {
            graph->setProg(newprog);
        }

        return modified;
    }

    Net2* net;
    Net2::Impl* netimpl;
    std::vector<int> usecounts;
    int64_t C0;
};

void Net2::Impl::constFold()
{
    ConstFolding constfolder(net);
    constfolder.process();
}

}}

