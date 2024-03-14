// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN2_TEST_ALL_HPP__
#define __OPENCV_DNN2_TEST_ALL_HPP__

#include <stdio.h>
#include <iostream>
#include "opencv2/dnn2.hpp"

namespace cv {
namespace dnn {

void test_0d();
void test_elemwise();
void test_flatten();
void test_reduce();
void test_tensor_basic();
void test_transform_layout();
void test_squeeze();
void test_unsqueeze();

}
}

#endif
