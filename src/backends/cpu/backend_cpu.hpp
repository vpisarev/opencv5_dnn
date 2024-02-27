// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_BACKEND_CPU_HPP__
#define __OPENCV_DNN_BACKEND_CPU_HPP__

namespace cv { namespace dnn {

struct CPUBackend : GraphBackend
{
    virtual ~CPUBackend();
    virtual Device* device() const CV_OVERRIDE;
    virtual std::string_view name() const CV_OVERRIDE;
    virtual bool supportType(int type) const CV_OVERRIDE;
    virtual int64_t preferredBlockSize(int type) const CV_OVERRIDE;
    virtual bool supportOp(const Op& op, const std::vector<SizeType>& inpst) const CV_OVERRIDE;
    virtual void preprocessGraph(Net2& net, const Graph& graph,
                        const std::vector<SizeType>& inpst,
                        std::vector<Buffer>& tempbufs) CV_OVERRIDE;
    virtual bool forward(Net2& net, const Graph& graph, std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs, std::vector<Buffer>& tempbufs) CV_OVERRIDE;
};

}}

#endif
