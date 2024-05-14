// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"

namespace cv { namespace dnn {

void Net2::Impl::initArgs()
{
    ArgInfo info;
    args.push_back(info);
    pattern_args.push_back(info);
    tensors.push_back(Tensor());
    bufidxs.push_back(-1);
}

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
    initialized = false;

    strm = &std::cout;
    dump_indent = 3;
}

Net2::Impl::~Impl() { clear(); }

void Net2::Impl::initialize()
{
    if (!initialized) {
        //constFold();
        //fuse();
        //useBlockLayout();
        //assignBuffers();
        initialized = true;
    }
}

void Net2::Impl::clear()
{
    modelFormat = DNN_MODEL_GENERIC;

    argnames = NamesHash();
    dimnames = NamesHash();
    dimnames_ = std::vector<std::string>();
    args = std::vector<ArgInfo>();
    tensors = std::vector<Tensor>();
    bufidxs = std::vector<int>();
    buffers = std::vector<Buffer>();
    mainGraph = Graph();
}

void Net2::Impl::forwardGraph(const Graph&)
{
    CV_Error(Error::StsNotImplemented, "");
}

void Net2::Impl::useCounts(std::vector<int>&) const
{
    CV_Error(Error::StsNotImplemented, "");
}

void Net2::Impl::updateUseCounts(std::vector<int>&, const Graph&) const
{
    CV_Error(Error::StsNotImplemented, "");
}

void Net2::Impl::checkArgs(const std::vector<Arg>&) const
{
    CV_Error(Error::StsNotImplemented, "");
}

void Net2::Impl::checkArg(Arg) const
{
    CV_Error(Error::StsNotImplemented, "");
}

void Net2::Impl::assignBuffers()
{
    CV_Error(Error::StsNotImplemented, "");
}

void Net2::Impl::fuse()
{
    CV_Error(Error::StsNotImplemented, "");
}

void Net2::Impl::constFold()
{
    CV_Error(Error::StsNotImplemented, "");
}

}}

