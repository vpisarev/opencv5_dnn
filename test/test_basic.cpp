// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_all.hpp"

namespace cv { namespace dnn {

void test_0d()
{
    Mat m(0, nullptr, CV_64F);
    Mat m2 = m;
    printf("m.dims=%d, m.rows=%d, m.cols=%d, m.type=%d, m.empty()=%d, m.total()=%zu\n", m2.dims, m2.rows, m2.cols, m2.type(), m2.empty(), m2.total());
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
    Op meanop = ElemwiseOp::create(ELWISE_MEAN);
    Op mulop = ElemwiseOp::create(ELWISE_MUL);
    meanop->forward(net, g, {a, b}, c, tmp);
    c[0].dump(std::cout, 0) << "\n";
    mulop->forward(net, g, {s, c[0]}, c, tmp);
    c[0].dump(std::cout, 0) << "\n";
    a = Tensor({{m, 1}, LAYOUT_UNKNOWN}, CV_32S);
    b = Tensor({{1, n}, LAYOUT_UNKNOWN}, CV_32S);
    adata = a.ptr<int>();
    bdata = b.ptr<int>();
    for (int i = 0; i < m; i++) adata[i] = i+1;
    for (int i = 0; i < n; i++) bdata[i] = i+1;
    mulop->forward(net, g, {a, b}, c, tmp);
    c[0].dump(std::cout, 0) << "\n";
    a = Tensor({{3, 3, 3}, LAYOUT_UNKNOWN}, CV_32F);
    float* adataf = a.ptr<float>();
    for (int i = 0; i < 3*3*3; i++)
        adataf[i] = CV_PI*0.03f*i;
    Op sinop = ElemwiseOp::create(ELWISE_SIN);
    sinop->forward(net, g, {a}, c, tmp);
    c[0].dump(std::cout, 0) << "\n";
}

void test_flatten()
{
    int N = 2, m = 3, n = 4;
    printf("=========== FLATTEN TEST ===========\n");
    Tensor a({{N, m, n}, LAYOUT_NCHW}, CV_32S);
    std::vector<Tensor> c0, c1 = {a};
    std::vector<Buffer> tmp;
    int* adata = a.ptr<int>();

    for (int i = 0; i < N*m*n; i++) {
        adata[i] = i;
    }
    Net2 net;
    Graph g = net.newGraph("main", {}, {}, true);

    std::cout << "input: ";
    a.dump(std::cout, 0) << "\n";
    std::cout << "==============================\n";

    Op op0 = FlattenOp::create();

    op0->forward(net, g, {a}, c1, tmp);
    std::cout << "output0 (after inplace Flatten {axis=1}): ";
    c1[0].dump(std::cout, 0) << "\n-----------------------\n";

    CV_Assert(a.handle() == c1[0].handle());

    op0->forward(net, g, {a}, c0, tmp);
    std::cout << "output0 (after Flatten {axis=1}): ";
    c0[0].dump(std::cout, 0) << "\n-----------------------\n";
}


void test_reduce()
{
    int N = 2, m = 3, n = 4;
    printf("=========== REDUCE OP TEST ===========\n");
    Tensor a({{N, m, n}, LAYOUT_UNKNOWN}, CV_32S);
    std::vector<int> a0 = {0}, a1 = {1}, aL={-1};
    Tensor axes[] = {
        Tensor::makeVector(a0),
        Tensor::makeVector(a1),
        Tensor::makeVector(aL),
        Tensor()
    };
    std::vector<Tensor> c;
    std::vector<Buffer> tmp;
    int* adata = a.ptr<int>();

    for (int i = 0; i < N*m*n; i++) {
        adata[i] = i;
    }
    Net2 net;
    Graph g = net.newGraph("main", {}, {}, true);
    Op reduce_ops[] = {
        ReduceOp::create(REDUCE_SUM, false),
        ReduceOp::create(REDUCE_MAX, false),
        ReduceOp::create(REDUCE_MIN, false)
    };

    std::cout << "input: ";
    a.dump(std::cout, 0) << "\n";
    std::cout << "==============================\n";

    for (int i = 0; i < 12; i++) {
        int opidx = i / 4;
        int aidx = i % 4;
        Op op = reduce_ops[opidx];
        Tensor axes_i = axes[aidx];
        op->forward(net, g, {a, axes_i}, c, tmp);
        std::cout << "op: " << op->name() << "\n";
        std::cout << "axes: ";
        axes_i.dump(std::cout, 0) << "\n";
        std::cout << "result: ";
        c[0].dump(std::cout, 0) << "\n";
        std::cout << "------------------------------\n";
    }
}


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
    t2.dump(std::cout, 3) << "\n";
}

void test_transform_layout()
{
    int N = 2, m = 3, n = 4;
    printf("=========== TRANSFORM LAYOUT TEST ===========\n");
    Tensor a({{N, m, n}, LAYOUT_NCHW}, CV_32S);
    std::vector<Tensor> c0, c1, c2, c3;
    std::vector<Buffer> tmp;
    int* adata = a.ptr<int>();

    for (int i = 0; i < N*m*n; i++) {
        adata[i] = i;
    }
    Net2 net;
    Graph g = net.newGraph("main", {}, {}, true);

    std::cout << "input: ";
    a.dump(std::cout, 0) << "\n";
    std::cout << "==============================\n";

    Op op0 = TransformLayoutOp::create(LAYOUT_NHWC);
    Op op1 = TransformLayoutOp::create(LAYOUT_NCHW);
    Op op2 = TransformLayoutOp::create(LAYOUT_NCHWc, 8);

    op0->forward(net, g, {a}, c0, tmp);
    std::cout << "output0 (NCHW=>NHWC): ";
    c0[0].dump(std::cout, 0) << "\n-----------------------\n";

    op1->forward(net, g, c0, c1, tmp);
    std::cout << "output1 (NHWC=>NCHW): ";
    c1[0].dump(std::cout, 0) << "\n-----------------------\n";

    op2->forward(net, g, c1, c2, tmp);
    std::cout << "output2 (NCHW=>NCHW8): ";
    c2[0].dump(std::cout, 0) << "\n-----------------------\n";

    op1->forward(net, g, c2, c3, tmp);
    std::cout << "output3 (NCHW8=>NCHW): ";
    c3[0].dump(std::cout, 0) << "\n-----------------------\n";
}

void test_squeeze()
{
    int N = 2, m = 3, n = 1;
    printf("=========== SQUEEZE TEST ===========\n");
    Tensor a({{N, m, n}, LAYOUT_NCHW}, CV_32S);
    std::vector<Tensor> c0;
    std::vector<Buffer> tmp;
    int* adata = a.ptr<int>();

    for (int i = 0; i < N*m*n; i++) {
        adata[i] = i;
    }
    Net2 net;
    Graph g = net.newGraph("main", {}, {}, true);

    std::cout << "input: ";
    a.dump(std::cout, 0) << "\n";
    std::cout << "==============================\n";

    Op op0 = SqueezeOp::create();

    op0->forward(net, g, {a, Tensor()}, c0, tmp);
    std::cout << "output0 (squeeze all 1's in the shape): ";
    c0[0].dump(std::cout, 0) << "\n-----------------------\n";
}

}}
