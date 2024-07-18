// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/net2_impl.hpp"

namespace cv { namespace dnn {

using std::vector;
using std::string;

/* Inserts layout conversion operations (if needed) into the model graph and subgraphs.

 Some of the operations
 (let's call them 'operations of category B' or B-operations, 'B' stands for 'Block'),
 most notably Convolution (including depth-wise convolution), MaxPool and AveragePool,
 can be computed more efficiently only if data is represented in so-called block layout,
 i.e. when 4D tensor NxCxHxW is represented in memory as 5D tensor NxC1xHxWxC0,
 where all C channels of the original tensor are split into C1 groups of C0 channels each.
 For each spatial location (y=y0, x=x0) each group of C0 channels is stored sequentially.
 C1 is thus computed as following: C1 = (C + C0-1)/C0, where division '/' is performed with truncation.

 Some other operations (let's call them A-operations, 'A' stands for 'Any'),
 most notably unary element-wise operations or special cases of binary
 element-wise operations can be easily computed in any layout, including block layout.

 Finally, all other operations (denoted as C-operations, 'C' for 'Casual' or 'Channels')
 does not support block layout at all. The inputs should come in the original format,
 depending on the model (e.g. NCHW in the case of Onnx).

 So we want to transform the graph so that:
 1. B-operations always take the inputs in block layout. If not, the inputs must be converted to
    block layout prior to B-operation. Note that only the first input should be converted
    in the case of Convolution, convolution weights (if constant) are pre-processed separately.
 2. C-operations always take the inputs in non-block layout, e.g. NCHW. If some of the inputs are
    stored in block layout, they must be converted from block layout prior to C-operation.
 3. the number of layout transformation operations is minimal,
    i.e. we don't do transformations unless it's necessary.

 Note that this graph transformation is applied after fusion, since inside a fused operation
 (e.g. 'Convolution + Batch Norm + Activation + Adding Skip connection') we don't need to
 transform layout. That is, we have to deal with less operations at this stage.
*/

struct BlockLayoutTransformer
{
    BlockLayoutTransformer(Net2* net_) : net(net_), netimpl(net_->impl()) {}

    Net2* net;
    Net2::Impl* netimpl;
    vector<TensorLayout> layouts; // layouts for each argument
    vector<Arg> blockCache; // if an Arg needs to be converted to block layout and then it's used by several operations,
                            // then we reuse once transformed arg, don't transform it several times
    vector<Arg> nonblockCache; // if an Arg needs to be converted to non-block layout and then it's used by several operations,
                               // then we reuse once transformed arg, don't transform it several times
    TensorLayout defaultLayout;

    std::pair<Arg,Node> getProperArg(const Arg& arg, bool block, int64_t defaultC0)
    {
        if (arg.empty() || (layouts[arg.idx] == LAYOUT_NCHWc) == block)
            return {arg, Node()};
        std::vector<Arg> *cache, *another_cache;
        if (block) {
            cache = &blockCache;
            another_cache = &nonblockCache;
        } else {
            cache = &nonblockCache;
            another_cache = &blockCache;
        }
        Arg cached = cache->at(arg.idx);
        if (!cached.empty())
            return {cached, Node()};
        const ArgInfo& info = net->argInfo(arg);
        cached = net->newArg(info.name + (block ? ".block" : ".nonblock"), DNN_ARG_TEMP);
        cache->at(arg.idx) = cached;
        CV_Assert(cached.idx == (int)layouts.size());
        Op tr_op;
        if (block)
            tr_op = TransformLayoutOp::create(LAYOUT_NCHWc, defaultC0);
        else
            tr_op = TransformLayoutOp::create(defaultLayout, 0);

        std::vector<Arg> inputs = {arg}, outputs = {cached};
        Node tr_node = NodeData::create(info.name + (block ? ".to_block" : ".from_block"), tr_op, inputs, outputs);

        layouts.push_back(block ? LAYOUT_NCHWc : defaultLayout);
        cache->push_back(Arg());
        another_cache->push_back(Arg());

        return {cached, tr_node};
    }

    void transformGraph(Graph& g)
    {
        const vector<Node>& curr_prog = g->prog();
        // [TODO] maybe we need to infer type and pass it to preferredBlockSize(), so that block size is dynamic
        int defaultType = CV_32F;
        int64_t defaultC0 = 8;//net->getBackend(0)->preferredBlockSize(defaultType);
        vector<Node> new_prog;
        std::vector<Arg> new_inputs;
        size_t nchanges = 0;

        for (const Node& node: curr_prog) {
            const Op& op = node->op();
            const vector<Arg>& inputs = node->inputs();
            const vector<Arg>& outputs = node->outputs();
            size_t ninputs = inputs.size(), noutputs = outputs.size();
            new_inputs.clear();
            int op_blockiness = 0;
            TensorLayout layout0 = defaultLayout;
            std::string_view op_name = op->name();
            std::string_view name = node->name();
            //std::cout << "name: " << name << ", op_name: " << op_name << ", inp0 layout: " << layoutToString(layouts[inputs[0].idx]) << "\n";

            for (const Graph& subgraph: node->subgraphs()) {
                transformGraph(const_cast<Graph&>(subgraph));
                nchanges++;
            }

            for (size_t i = 0; i < ninputs; i++) {
                int blockiness = op->supportBlockLayout((int)i, (int)ninputs);
                if (blockiness > 0) {
                    CV_Assert(op_blockiness >= 0);
                    op_blockiness = 1;
                } else if (blockiness < 0 && op_blockiness <= 0) {
                    op_blockiness = -1;
                }

                Arg inp = inputs[i], new_inp = inp;
                if (i == 0)
                    layout0 = layouts[inp.idx];
                if (blockiness != 0) {
                    auto new_inp_node = getProperArg(inp, blockiness > 0, defaultC0);
                    new_inp = new_inp_node.first;
                    if (new_inp_node.second) {
                        new_prog.push_back(new_inp_node.second);
                    }
                    nchanges += !(new_inp == inp);
                }
                new_inputs.push_back(new_inp);
            }

            TensorLayout layout1 = op_blockiness > 0 ? LAYOUT_NCHWc : op_blockiness < 0 ? defaultLayout : layout0;
            for (size_t i = 0; i < noutputs; i++) {
                Arg out = outputs[i];
                layouts[out.idx] = layout1;
            }
            Node new_node = NodeData::create(node->name(), op, new_inputs, outputs, node->subgraphs());
            new_prog.push_back(new_node);
        }
        
        if (nchanges > 0)
            g->setProg(new_prog);
    }

    void transform()
    {
        size_t nargs = netimpl->args.size();
        defaultLayout = netimpl->defaultLayout;

        layouts.assign(nargs, defaultLayout);
        blockCache.assign(nargs, Arg());
        nonblockCache.assign(nargs, Arg());

        transformGraph(netimpl->mainGraph);
    }
};

void Net2::Impl::useBlockLayout()
{
    BlockLayoutTransformer use_block_layout(net);
    use_block_layout.transform();
}

}}

