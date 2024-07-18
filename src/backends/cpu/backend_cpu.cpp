// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "../../engine/net2_impl.hpp"
#include "backend_cpu.hpp"

namespace cv { namespace dnn {

static CPUBackend dnnCpuBackend;

GraphBackend::~GraphBackend()
{
}

GraphBackend* GraphBackend::fromSpec(std::string_view backendSpec)
{
    if (backendSpec == "cpu" || backendSpec == "CPU" || backendSpec == "default")
        return &dnnCpuBackend;
    CV_Error(Error::StsNotImplemented, format("unknown backend '%s'", std::string(backendSpec).c_str()));
    return nullptr;
}

void GraphBackend::preprocessGraph(Graph& graph,
                    const std::vector<SizeType>& inpst,
                    std::vector<Buffer>&) const
{
    CV_Assert(graph && graph->inputs().size() == inpst.size());

    // don't do anything with the graph by default
}

bool GraphBackend::forward(Graph& graph, std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs, std::vector<Buffer>& tempbufs) const
{
    CV_Error(Error::StsNotImplemented, "");
    return false;
}

CPUBackend::~CPUBackend() {}
Device* CPUBackend::device() const { return Device::CPU(); }
std::string_view CPUBackend::name() const { return "org.opencv.dnn.backend.cpu"; }
bool CPUBackend::supportType(int type) const
{
    if (checkHardwareSupport(CV_CPU_FP16))
        return type != CV_16BF;
    return type != CV_16F && type != CV_16BF;
}

int64_t CPUBackend::preferredBlockSize(int) const
{
    // [TODO]
    return 8;
}

bool CPUBackend::supportOp(const Op&, const std::vector<SizeType>&) const
{
    // CPU backend must support everything
    return true;
}

void CPUBackend::preprocessGraph(Graph& graph,
                    const std::vector<SizeType>& inpst,
                    std::vector<Buffer>& tempbufs) const
{
    GraphBackend::preprocessGraph(graph, inpst, tempbufs);
}

bool CPUBackend::forward(Graph& graph, std::vector<Tensor>& inputs,
             std::vector<Tensor>& outputs, std::vector<Buffer>& tempbufs) const
{
    return GraphBackend::forward(graph, inputs, outputs, tempbufs);
}

}}
