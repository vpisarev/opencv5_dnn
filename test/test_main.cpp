#include <stdio.h>
#include <iostream>
#include "opencv2/dnn2.hpp"

using namespace cv;
using namespace cv::dnn;

int main(int, char**)
{
    Tensor t({{1, 1, 15, 10}, LAYOUT_NCHW}, CV_32F);
    CV_Assert(!t.empty());
    CV_Assert(t.total() == 150);
    CV_Assert(t.type() == CV_32F);
    t.setTo(1);
    Tensor t2;
    t.copyTo(t2);
    CV_Assert(t2.total() == t.total());
    CV_Assert(t2.type() == t.type());
    CV_Assert(t.buffer() && t2.buffer() && t.buffer() != t2.buffer());
    CV_Assert(t2.deviceType() == Device_CPU);
    float* data = t2.ptr<float>();
    size_t sz = t2.total();
    for (size_t i = 0; i < sz; i++) {
        data[i] = (float)(data[i] + i);
    }
    t2.dump(std::cout, 0);
    return 0;
}
