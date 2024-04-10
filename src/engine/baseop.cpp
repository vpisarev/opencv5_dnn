// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"

namespace cv { namespace dnn {

BaseOp::~BaseOp() {}
std::string_view BaseOp::origName() const { return name(); }
std::string_view BaseOp::profileName() const { return name(); }
std::ostream& BaseOp::dumpAttrs(std::ostream& strm, int) const { return strm; }

void BaseOp::dumpTensorAttr(std::ostream& strm, std::string_view name,
                            const Tensor& t, int indent)
{
    prindent(strm, indent);
    strm << '\"' << name << "\": ";
    t.dump(strm, indent + 3);
}

void BaseOp::dumpScalarAttr(std::ostream& strm, std::string_view name,
                            int type, const void* scalar, int indent)
{
    prindent(strm, indent);
    strm << '\"' << name << "\": ";
    if (type == CV_8U)
        strm << *(const uint8_t*)scalar;
    else if (type == CV_8S)
        strm << *(const int8_t*)scalar;
    else if (type == CV_16U)
        strm << *(const uint16_t*)scalar;
    else if (type == CV_16S)
        strm << *(const int16_t*)scalar;
    else if (type == CV_32U)
        strm << *(const uint32_t*)scalar;
    else if (type == CV_32S)
        strm << *(const int32_t*)scalar;
    else if (type == CV_64U)
        strm << *(const uint64_t*)scalar;
    else if (type == CV_64S)
        strm << *(const int64_t*)scalar;
    else if (type == CV_32F)
        strm << *(const float*)scalar;
    else if (type == CV_64F)
        strm << *(const double*)scalar;
    else if (type == CV_16F)
        strm << (float)*(const cv::float16_t*)scalar;
    else if (type == CV_16BF)
        strm << (float)*(const cv::bfloat16_t*)scalar;
    else if (type == CV_Bool)
        strm << (*(const bool*)scalar ? "true" : "false");
    else {
        CV_Error(Error::StsNotImplemented, "");
    }
}

void BaseOp::dumpStringAttr(std::ostream& strm, std::string_view name,
                            std::string_view value, int indent)
{
    prindent(strm, indent);
    strm << '\"' << name << "\": \"" << value << '\"';
}

void BaseOp::setProfileEntry(int idx)
{
    profileIdx = idx;
}

int BaseOp::getProfileEntry() const
{
    return profileIdx;
}

bool BaseOp::alwaysSupportInplace() const
{
    return false;
}

bool BaseOp::supportInplace(const Net2&, const Graph&,
                    const std::vector<Arg>&,
                    const std::vector<SizeType>&) const
{
    return alwaysSupportInplace();
}

int64_t BaseOp::getFLOPS(const std::vector<SizeType>&,
                 const std::vector<SizeType> &outputs) const
{
    if (outputs.empty())
        return 0;
    return (int64_t)outputs[0].size.total();
}

void BaseOp::forward(Net2& net, Graph& graph,
                    const std::vector<Tensor>& inputs,
                    Tensor& output,
                    std::vector<Buffer>& tempbufs)
{
    std::vector<Tensor> outputs = {output};
    forward(net, graph, inputs, outputs, tempbufs);
    CV_Assert(outputs.size() == 1);
    output = outputs[0];
}

}}
