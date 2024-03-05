// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

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

void test_elemwise()
{
    int m = 5, n = 6;
    printf("=========== ELEMWISE OP TEST ===========\n");
    Tensor a({{m, n}, LAYOUT_UNKNOWN}, CV_32S);
    Tensor b({{m, n}, LAYOUT_UNKNOWN}, CV_32S);
    Tensor s = Tensor::makeScalar(3);
    std::vector<Tensor> c;//({{m, n}, LAYOUT_UNKNOWN}, CV_32S);
    std::vector<Buffer> tmp;
    int* adata = a.ptr<int>();
    int* bdata = b.ptr<int>();

    for (int i = 0; i < m*n; i++) {
        adata[i] = -i;
        bdata[i] = i*3;
    }
    Net2 net;
    Graph g = net.newGraph("main", {}, {}, true);
    Op addop = ElemwiseOp::create(ELWISE_ADD);
    Op mulop = ElemwiseOp::create(ELWISE_MUL);
    addop->forward(net, g, {a, b}, c, tmp);
    c[0].dump(std::cout, 0);
    std::cout << "\n";
    mulop->forward(net, g, {s, c[0]}, c, tmp);
    c[0].dump(std::cout, 0);
    a = Tensor({{m, 1}, LAYOUT_UNKNOWN}, CV_32S);
    b = Tensor({{1, n}, LAYOUT_UNKNOWN}, CV_32S);
    adata = a.ptr<int>();
    bdata = b.ptr<int>();
    for (int i = 0; i < m; i++) adata[i] = i+1;
    for (int i = 0; i < n; i++) bdata[i] = i+1;
    mulop->forward(net, g, {a, b}, c, tmp);
    std::cout << "\n";
    c[0].dump(std::cout, 0);
}

}}
