// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "engine/engine.hpp"

namespace cv { namespace dnn {

bool isIntType(int type)
{
    int depth = CV_MAT_DEPTH(type);
    return depth < CV_32F || depth == CV_32U || depth == CV_64U || depth == CV_64S;
}

bool isSignedIntType(int type)
{
    int depth = CV_MAT_DEPTH(type);
    return depth == CV_8S || depth == CV_16S || depth == CV_32S || depth == CV_64S;
}

bool isFPType(int type)
{
    int depth = CV_MAT_DEPTH(type);
    return depth == CV_32F || depth == CV_64F || depth == CV_16F || depth == CV_16BF;
}

int normalizeAxis(int axis, int ndims)
{
    if (axis < 0)
        axis += ndims;
    CV_Assert(0 <= axis && axis < ndims);
    return axis;
}

// extract array of axes (up to TensorSize::MAX_DIMS).
// Each axis should be within a range [-ndims, ndims-1]
int normalizeAxes(const Tensor& axes, int ndims, int* axisbuf, bool* axismask_)
{
    bool axismaskbuf[TensorSize::MAX_DIMS];
    bool* axismask = axismask_ ? axismask_ : axismaskbuf;

    if (axes.empty()) {
        for (int i = 0; i < ndims; i++)
            axismask[i] = false;
        return 0;
    }

    int axistype = axes.type();
    CV_Assert(axes.ndims() <= 1);
    CV_Assert(axistype == CV_32S || axistype == CV_64S);
    int naxes = (int)axes.total();
    const int32_t* axes32 = (const int32_t*)axes.data();
    const int64_t* axes64 = (const int64_t*)axes.data();

    CV_Assert(naxes <= ndims);

    for (int i = 0; i < ndims; i++)
        axismask[i] = false;

    for (int i = 0; i < (int)naxes; i++) {
        int axis = axistype == CV_32S ? (int)axes32[i] : axes64 ? (int)axes64[i] : i;
        axis = normalizeAxis(axis, ndims);
        if (axismask[axis]) {
            CV_Error(Error::StsError, "there are duplicated axes in the axes specification");
        }
        axismask[axis] = true;
        axisbuf[i] = axis;
    }

    return (int)naxes;
}

}}
