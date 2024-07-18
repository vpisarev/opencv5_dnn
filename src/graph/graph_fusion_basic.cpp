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

struct ModelFusionBasic
{
    ModelFusionBasic(Net2* net_) : net(net_), netimpl(net_->impl()) {}

    void fuse()
    {
        int i, niter = 10;
        netimpl->useCounts(usecounts);
        for (i = 0; i < niter; i++) {
            bool fused_any = fuseGraph(netimpl->mainGraph);
            if (!fused_any)
                break;
        }
    }

    bool isConstScalarTensor(Arg arg, float* compare_with) const
    {
        const Tensor* t;
        void* data;
        if (!net->isConstArg(arg))
            return false;
        t = &netimpl->tensors.at(arg.idx);
        data = t->data();
        if (t->total() != 1 || t->type() != CV_32F)
            return false;
        if (compare_with && *(float*)data != *compare_with)
            return false;
        return true;
    }

    Node* getNode(std::vector<Node>& newprog, int op_idx) const
    {
        return op_idx >= 0 ? &newprog.at(op_idx) : 0;
    }

    template<typename _OpType> _OpType* getOp(const Node* n) const
    {
        return !n ? nullptr : dynamic_cast<_OpType*>((*n)->op().get());
    }

    template<typename _OpType>
    int_arg_pair isUnary(std::vector<Node>& newprog, const std::vector<int>& producer_of,
                         int t_inp, bool check_used_once) const
    {
        int op_idx;
        const Node* node;
        if (t_inp < 0) return std::make_pair(-1, Arg());
        op_idx = producer_of.at(t_inp);
        node = getNode(newprog, op_idx);
        if (!node || (*node)->ninputs() != 1 || !getOp<_OpType>(node) ||
            (check_used_once && usecounts.at((*node)->inputs(0).idx) != 1))
            return std::make_pair(-1, Arg());
        return std::make_pair(op_idx, (*node)->inputs(0));
    }

    bool fuseGraph(Graph& graph)
    {
        vector<Arg> removed_args;
        bool modified = false;
        const std::vector<Node>& prog = graph->prog();
        size_t i, nargs = netimpl->args.size(), nops = prog.size();
        std::vector<int> producer_of(nargs, -1);
        std::vector<Node> newprog;
        std::vector<Arg> fused_inputs;

        for (i = 0; i < nops; i++) {
            const Node& node = prog[i];
            std::vector<Graph>& subgraphs = const_cast<std::vector<Graph>&>(node->subgraphs());
            for (Graph& g: subgraphs) {
                if (fuseGraph(g))
                    modified = true;
            }
            const std::vector<Arg>& inputs = node->inputs();
            const std::vector<Arg>& outputs = node->outputs();
            size_t ninputs = inputs.size();
            Op fused_op = node->op();
            int fused_node_idx = -1;
            removed_args.clear();
            fused_inputs.clear(); // leave it empty in the merge patterns below to re-use original fused node inputs as-is.

            for(;;) {
                BatchNormOp* bn = getOp<BatchNormOp>(&node);
                ElemwiseOp* elemwise = getOp<ElemwiseOp>(&node);

                // merge convolution and batch norm
                if (bn && ninputs == 1 &&
                    usecounts.at(inputs[0].idx) == 1) {
                    Arg bn_inp = inputs[0];
                    int conv_node_idx = producer_of.at(bn_inp.idx);
                    Node* conv_node = getNode(newprog, conv_node_idx);
                    ConvOp* conv_op = getOp<ConvOp>(conv_node);
                    if (conv_op && (*conv_node)->ninputs() == 1) {
                        bool ok = conv_op->fuseBatchNorm(node->op());
                        if (ok) {
                            fused_node_idx = conv_node_idx;
                            fused_op = (*conv_node)->op();
                            removed_args.push_back(bn_inp);
                            break;
                        }
                    }
                }

                // merge residual 'add' into 'conv' node
                if (elemwise && elemwise->opcode == ELWISE_ADD && ninputs == 2) {
                    ArgInfo& info0 = netimpl->args[inputs[0].idx];
                    ArgInfo& info1 = netimpl->args[inputs[1].idx];

                    if (info0.type == info1.type && info0.size == info1.size) {
                        int op0 = producer_of.at(inputs[0].idx);
                        int op1 = producer_of.at(inputs[1].idx);
                        int conv_node_idx;
                        Arg residual, conv_out;

                        if (op0 > op1) { // choose the latter op to ensure that the other component is already computed
                            conv_node_idx = op0;
                            conv_out = inputs[0];
                            residual = inputs[1];
                        } else {
                            conv_node_idx = op1;
                            conv_out = inputs[1];
                            residual = inputs[0];
                        }

                        Node* conv_node = getNode(newprog, conv_node_idx);
                        ConvOp* conv_op = getOp<ConvOp>(conv_node);
                        if (conv_op && !conv_op->activ && !conv_op->add_residual && usecounts[conv_out.idx] == 1) {
                            conv_op->add_residual = true;
                            fused_node_idx = conv_node_idx;
                            fused_op = (*conv_node)->op();
                            const std::vector<Arg>& conv_inputs = (*conv_node)->inputs();
                            fused_inputs.assign(conv_inputs.begin(), conv_inputs.end());
                            fused_inputs.push_back(residual);
                            removed_args.push_back(conv_out);
                            break;
                        }
                    }
                }

                // merge convolution and activation
                if (elemwise && ninputs == 1 &&
                    usecounts.at(inputs[0].idx) == 1) {
                    Arg activ_inp = inputs[0];
                    int conv_node_idx = producer_of.at(activ_inp.idx);
                    Node* conv_node = getNode(newprog, conv_node_idx);
                    ConvOp* conv_op = getOp<ConvOp>(conv_node);
                    if (conv_op) {
                        bool ok = conv_op->fuseActivation(node->op());
                        if (ok) {
                            fused_node_idx = conv_node_idx;
                            fused_op = (*conv_node)->op();
                            removed_args.push_back(activ_inp);
                            break;
                        }
                    }
                }

                break;
            }

            if (fused_node_idx >= 0) {
                modified = true;
                const Node& orig_node = newprog.at(fused_node_idx);
                if (fused_inputs.empty()) {
                    const std::vector<Arg>& orig_inputs = orig_node->inputs();
                    fused_inputs.assign(orig_inputs.begin(), orig_inputs.end());
                }
                Node fused_node = NodeData::create(orig_node->name(), fused_op,
                                                   fused_inputs, outputs,
                                                   orig_node->subgraphs());
                newprog.at(fused_node_idx) = fused_node;
                for (Arg new_out: outputs)
                    producer_of[new_out.idx] = fused_node_idx;
                for (Arg old_out: removed_args) {
                    usecounts.at(old_out.idx) = 0;
                    producer_of.at(old_out.idx) = -1;
                }
            } else {
                for (auto out: outputs)
                    producer_of[out.idx] = (int)newprog.size();
                newprog.push_back(node);
            }
        }

        if (modified) {
            size_t i, j = 0, nops = newprog.size();
            for (i = 0; i < nops; i++) {
                if (newprog[i]->op()) {
                    if (j < i)
                        newprog[j] = newprog[i];
                    j++;
                }
            }
            newprog.resize(j);
            printf("fused some ops in graph %s. size before: %zu ops, size after: %zu ops\n",
                   graph->name().data(), nops, j);
            graph->setProg(newprog);
        }

        return modified;
    }

    Net2* net;
    Net2::Impl* netimpl;
    vector<int> usecounts;
};

void Net2::Impl::fuse()
{
    ModelFusionBasic basicFusion(net);
    basicFusion.fuse();
}

}}
