// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_all.hpp"

namespace cv { namespace dnn {

void test_reshape()
{
    int N = 2, m = 3, n = 4;
    printf("=========== RESHAPE TEST ===========\n");
    Tensor a({{N, m, n}, LAYOUT_UNKNOWN}, CV_32S);
    std::vector<int> shapevec = {0, 1, -1};
    Tensor shape = Tensor::makeVector(shapevec);
    std::vector<Tensor> c0, c1 = {a};
    std::vector<Buffer> tmp;
    int* adata = a.ptr<int>();

    for (int i = 0; i < N*m*n; i++) {
        adata[i] = i;
    }
    Net2 net;
    Graph g = net.newGraph("main", {}, {});
    Op op = ReshapeOp::create(false);

    std::cout << "input: ";
    a.dump(std::cout, 0) << "\n";
    std::cout << "==============================\n";

    op->forward(net, g, {a, shape}, c0, tmp);
    std::cout << "result (shape = ";
    c0[0].sizetype().dump(std::cout) << "): ";
    c0[0].dump(std::cout, 0) << "\n";

    op->forward(net, g, {a, shape}, c1, tmp);
    std::cout << "result after inplace reshape (shape = ";
    c1[0].sizetype().dump(std::cout) << "): ";
    c1[0].dump(std::cout, 0) << "\n";
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
    Graph g = net.newGraph("main", {}, {});

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
    Graph g = net.newGraph("main", {}, {});

    std::cout << "input: ";
    a.dump(std::cout, 0) << "\n";
    std::cout << "==============================\n";

    Op op0 = SqueezeOp::create();

    op0->forward(net, g, {a}, c0, tmp);
    std::cout << "output0 (squeeze all 1's in the shape): ";
    c0[0].dump(std::cout, 0) << "\n-----------------------\n";
}

void test_unsqueeze()
{
    int N = 2, m = 3;
    printf("=========== UNSQUEEZE TEST ===========\n");
    Tensor a({{N, m}, LAYOUT_NCHW}, CV_32S);
    std::vector<int> axesbuf = {0, -1};
    Tensor axes = Tensor::makeVector(axesbuf);
    std::vector<Tensor> c0;
    std::vector<Buffer> tmp;
    int* adata = a.ptr<int>();

    for (int i = 0; i < N*m; i++) {
        adata[i] = i;
    }
    Net2 net;
    Graph g = net.newGraph("main", {}, {});

    std::cout << "input: ";
    a.dump(std::cout, 0) << "\n";
    std::cout << "==============================\n";

    Op op0 = UnsqueezeOp::create();

    op0->forward(net, g, {a, axes}, c0, tmp);
    std::cout << "output0 (unsqueeze): shape=";
    c0[0].sizetype().dump(std::cout) << ", data: ";
    c0[0].dump(std::cout, 0) << "\n-----------------------\n";
}

}}
