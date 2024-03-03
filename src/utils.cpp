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

}}
