// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN2_TEST_UTILS_HPP__
#define __OPENCV_DNN2_TEST_UTILS_HPP__

#include <stdio.h>
#include <iostream>
#include "opencv2/dnn2.hpp"

namespace cv {
namespace dnn {

TensorSize ref_conv_infer_shapes(const TensorSize& inpsize,
                                 const ConvParams& convparams,
                                 const TensorSize& wsize=TensorSize());

}
}

#endif

