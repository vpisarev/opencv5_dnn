// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "../../engine/engine.hpp"
#include "backend_cpu.hpp"

namespace cv { namespace dnn {



CPUBackend::~CPUBackend() {}
Device* CPUBackend::device() const { return Device::CPU(); }
std::string_view CPUBackend::name() const { return "org.opencv.dnn.backend.cpu"; }
bool CPUBackend::supportType(int type) const
{
#if CV_16F
    return type != CV_16BF;
#else
    return type != CV_16F && type != CV_16BF;
#endif
}

int64_t CPUBackend::preferredBlockSize(int) const
{
    // [TODO]
    return 1;
}

bool CPUBackend::supportOp(const Op&, const std::vector<SizeType>&) const
{
    // CPU backend must support everything
    return true;
}

}}
