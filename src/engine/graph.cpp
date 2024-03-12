// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"

namespace cv { namespace dnn {

void prindent(std::ostream& strm, int indent)
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

std::ostream& NodeData::dump(const Net2& net, std::ostream& strm,
                             int indent, bool comma) const
{
    size_t ninputs = inputs_.size(), noutputs = outputs_.size();
    size_t ngraphs = subgraphs_.size();
    int delta_indent = net.indent();
    int subindent = indent + delta_indent;
    int argindent = subindent + delta_indent;
    prindent(strm, indent);
    std::string_view opname = op_->name();
    strm << opname << "{\n";
    prindent(strm, subindent);
    strm << "name: \"" << name_ << "\",\n";
    op_->dumpAttrs(strm, subindent);
    prindent(strm, subindent);
    strm << "inputs: [\n";
    for (size_t i = 0; i < ninputs; i++) {
        net.dumpArg(strm, inputs_[i], argindent, i+1 < ninputs);
    }
    prindent(strm, subindent);
    strm << "],\n";
    prindent(strm, subindent);
    strm << "outputs: [\n";
    for (size_t i = 0; i < ninputs; i++) {
        net.dumpArg(strm, outputs_[i], argindent, i+1 < noutputs);
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
                     format("unsupported operation '%s' with subgraphs",
                            std::string(opname).c_str()));
        }
        for (size_t i = 0; i < ngraphs; i++) {
            prindent(strm, subindent);
            strm << names[i] << ": ";
            subgraphs_[i]->dump(strm, argindent, i+1 < ngraphs);
        }
    }
    prindent(strm, indent);
    strm << '}';
    if (comma)
        strm << ',';
    strm << '\n';
    return strm;
}


std::string_view NodeData::name() const { return name_; }
Op NodeData::op() const { return op_; }
const std::vector<Arg>& NodeData::inputs() const { return inputs_; }
const std::vector<Arg>& NodeData::outputs() const { return outputs_; }
const std::vector<Graph>& NodeData::subgraphs() const { return subgraphs_; }

GraphData::GraphData(const Net2& net, std::string_view name,
                     const std::vector<Arg>& inputs,
                     bool ispattern)
{
    ispattern_ = ispattern;
    net_ = (Net2*)&net;
    name_ = name;
    inputs_ = inputs;
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
    Graph g = std::make_shared<GraphData>((newnet ? *newnet : *net_), name_, inputs_, ispattern_);
    g->outputs_ = outputs_;
    g->backend_ = backend_;
    // don't copy optigraph_. It has to be re-created
    for (auto n : prog_) {
        g->prog_.push_back(n->clone(g->net_));
    }
    return g;
}

void GraphData::append(std::string_view node_name, const Op& op,
                       const std::vector<std::string_view>& outnames,
                       const std::vector<Arg>& inputs,
                       std::vector<Arg>& outputs)
{

}

Arg GraphData::append(std::string_view node_name, const Op& op,
                      std::string_view outname,
                      const std::vector<Arg>& inputs)
{

}

bool GraphData::isPattern() const { return ispattern_; }

std::ostream& GraphData::dump(std::ostream& strm, int indent, bool comma)
{
    size_t ninputs = inputs_.size(), noutputs = outputs_.size();
    int delta_indent = net_ ? net_->indent() : 3;
    int subindent = indent + delta_indent;
    int argindent = subindent + delta_indent;
    strm << "graph {\n";
    prindent(strm, subindent);
    strm << "name: ";
    if (name_.empty())
        strm << "<noname>\n";
    else
        strm << '\"' << name_ << "\"\n";
    prindent(strm, subindent);
    strm << "inputs: [\n";
    for (size_t i = 0; i < ninputs; i++) {
        net_->dumpArg(strm, inputs_[i], argindent, i+1 < ninputs);
    }
    prindent(strm, subindent);
    strm << "],\n";
    prindent(strm, subindent);
    strm << "outputs: [\n";
    for (size_t i = 0; i < ninputs; i++) {
        net_->dumpArg(strm, outputs_[i], argindent, i+1 < noutputs);
    }
    prindent(strm, subindent);
    strm << "],\n";
    prindent(strm, subindent);
    strm << "nodes: [\n";
    size_t nnodes = prog_.size();
    for (size_t i = 0; i < nnodes; i++) {
        prindent(strm, subindent);
        strm << "// op #" << i << "\n";
        const Node& node = prog_[i];
        node->dump(*net_, strm, argindent, i+1 < nnodes);
    }
    prindent(strm, subindent);
    strm << "]\n";
    prindent(strm, indent);
    strm << '}';
    if (comma)
        strm << ',';
    strm << '\n';
    return strm;
}

void GraphData::inferShapes(const std::vector<SizeType>& inpst,
                            std::vector<SizeType>& outst) const
{
}

Net2* GraphData::net() const { return net_; }
const std::vector<Arg>& GraphData::inputs() const { return inputs_; }
const std::vector<Arg>& GraphData::outputs() const { return outputs_; }
void GraphData::setOutputs(const std::vector<Arg>& outputs) {
    net_->checkArgs(outputs);
    outputs_ = outputs;
}
const std::vector<Node>& GraphData::prog() const { return prog_; }

OptimizedGraph GraphData::getOptimized() const { return optigraph_; }

void GraphData::setOptimized(const OptimizedGraph& optigraph)
{
    optigraph_ = optigraph;
}

}}
