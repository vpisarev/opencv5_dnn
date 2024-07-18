// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_all.hpp"
#include <math.h>

namespace cv { namespace dnn {

using std::string;

void test_import_onnx()
{
    const char* model_root = getenv("OPENCV5_DNN_MODEL_PATH");
    if (model_root == 0 || strlen(model_root) == 0) {
        model_root = ".";
    }
    string model_path = model_root + string("/resnet50-v1-12.onnx");
    OnnxReaderParams params;
    Net2 net = readNetFromONNX2(model_path, params);
    net.dump();
}

}}
