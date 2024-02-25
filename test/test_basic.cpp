#include "test_all.hpp"

namespace cv { namespace dnn {

void test_tensor_basic()
{
    printf("=========== BASIC_TENSOR TEST ===========\n");
    Tensor t({{1, 1, 5, 3}, LAYOUT_NCHW}, CV_32FC3);
    CV_Assert(!t.empty());
    CV_Assert(t.total() == 15);
    CV_Assert(t.type() == CV_32FC3);
    t.setTo(1.5);
    Tensor t2;
    t.copyTo(t2);
    CV_Assert(t2.total() == t.total());
    CV_Assert(t2.type() == t.type());
    CV_Assert(t.buffer() && t2.buffer() && t.buffer() != t2.buffer());
    CV_Assert(t2.deviceType() == Device_CPU);
    float* data = t2.ptr<float>();
    size_t sz = t2.total()*t2.channels();
    for (size_t i = 0; i < sz; i++) {
        data[i] = (float)(data[i] + i);
    }
    t2.dump(std::cout, 3);
}

}}
