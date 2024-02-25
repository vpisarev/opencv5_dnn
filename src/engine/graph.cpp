// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"

namespace cv { namespace dnn {

static void prindent(std::ostream& strm, int indent)
{
    for (int i = 0; i < indent; i++)
        strm << ' ';
}

NodeData::NodeData()
{
}

NodeData::NodeData(std::string_view name, const Op& op,
                   const std::vector<Arg>& inputs,
                   const std::vector<Arg>& outputs,
                   const std::vector<Graph>& subgraphs)
{
    name_ = name;
    op_ = op;
    inputs_ = inputs;
    outputs_ = outputs;
    subgraphs_ = subgraphs;
}

NodeData::~NodeData()
{
}

Node NodeData::clone(Net2* newnet) const
{
    Node node = std::make_shared<NodeData>(name_, op_, inputs_, outputs_);
    for (auto g: subgraphs_) {
        node->subgraphs_.push_back(g->clone(newnet));
    }
    return node;
}

void NodeData::dump(const Net2& net, std::ostream& strm,
                    int indent, size_t maxsz_small, bool comma) const
{
    size_t ninputs = inputs_.size(), noutputs = outputs_.size();
    size_t ngraphs = subgraphs_.size();
    int delta_indent = net.deltaIndent();
    int subindent = indent + delta_indent;
    int argindent = subindent + delta_indent;
    prindent(strm, indent);
    std::string_view opname = op_->name();
    strm << opname << "{\n";
    prindent(strm, subindent);
    strm << "name: \"" << name_ << "\",\n";
    op_->dumpAttrs(strm, subindent, maxsz_small);
    prindent(strm, subindent);
    strm << "inputs: [\n";
    for (size_t i = 0; i < ninputs; i++) {
        net.dumpArg(strm, inputs_[i], argindent, maxsz_small, i+1 < ninputs);
    }
    prindent(strm, subindent);
    strm << "],\n";
    prindent(strm, subindent);
    strm << "outputs: [\n";
    for (size_t i = 0; i < ninputs; i++) {
        net.dumpArg(strm, outputs_[i], argindent, maxsz_small, i+1 < noutputs);
    }
    prindent(strm, subindent);
    strm << "],\n";

    if (!subgraphs_.empty()) {
        size_t ngraphs = subgraphs_.size();
        std::vector<std::string_view> names;
        if (opname == "If")
            names = {"then", "else"};
        else if (opname == "Loop")
            names = {"body"};
        else {
            CV_Error(Error::StsError,
                     format("unsupported operation '%s' with subgraphs", std::string(opname).c_str()));
        }
        for (size_t i = 0; i < ngraphs; i++) {
            prindent(strm, subindent);
            strm << names[i] << ": ";
            subgraphs_[i]->dump(strm, argindent, maxsz_small, i+1 < ngraphs);
        }
    }
    prindent(strm, indent);
    if (comma)
        strm << ',';
    strm << '\n';
}


std::string_view NodeData::name() const { return name_; }
Op NodeData::op() const { return op_; }
const std::vector<Arg>& NodeData::inputs() const { return inputs_; }
const std::vector<Arg>& NodeData::outputs() const { return outputs_; }
const std::vector<Graph>& NodeData::subgraphs() const { return subgraphs_; }

GraphData::GraphData()
{
    ispattern_ = false;
    net_ = 0;
    backend_ = 0;
}

GraphData::GraphData(Net2& net, std::string_view name,
                     const std::vector<Arg>& inputs,
                     const std::vector<Arg>& outputs,
                     bool ispattern)
{
    ispattern_ = ispattern;
    net_ = &net;
    name_ = name;
    inputs_ = inputs;
    outputs_ = outputs;
    backend_ = 0;
}

GraphData::~GraphData()
{
}

std::string_view GraphData::name() const { return name_; }
bool GraphData::empty() const { return prog_.empty(); }
void GraphData::clear()
{
    prog_.clear();
}

Graph GraphData::clone(Net2* newnet) const
{
    Graph g = std::make_shared<GraphData>((newnet ? *newnet : *net_), name_, inputs_, outputs_, ispattern_);
    g->backend_ = backend_;
    // don't copy optigraph_. It has to be re-created
    for (auto n : prog_) {
        g->prog_.push_back(n->clone(g->net_));
    }
    return g;
}

void GraphData::append(std::string_view node_name, const Op& op,
                       const std::vector<Arg>& inputs,
                       const std::vector<std::string_view>& outnames,
                       std::vector<Arg>& outputs)
{

}

Arg GraphData::append(std::string_view node_name, const Op& op,
                      const std::vector<Arg>& inputs,
                      std::string_view outname)
{

}

bool GraphData::isPattern() const { return ispattern_; }
void GraphData::dump(std::ostream& strm, int indent, size_t maxsz_small, bool comma)
{
    
}

void GraphData::inferShapes(const std::vector<SizeType>& inpst,
                            std::vector<SizeType>& outst) const
{
}

Net2* GraphData::net() const { return net_; }
const std::vector<Arg>& GraphData::inputs() const { return inputs_; }
const std::vector<Arg>& GraphData::outputs() const { return outputs_; }
const std::vector<Node>& GraphData::prog() const { return prog_; }

}}
