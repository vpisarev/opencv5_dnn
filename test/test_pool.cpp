// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_all.hpp"
#include <math.h>

namespace cv { namespace dnn {

static void ref_globavgpool(const Tensor& inp, Tensor& out)
{
    TensorSize size = inp.size();
    int64_t N = size.size[0], C1 = size.size[1], H = size.size[2], W = size.size[3], C0 = size.size[4];
    double scale = H*W > 0 ? 1./(H*W) : 0;
    for (int64_t nc = 0; nc < N*C1*C0; nc++) {
        int64_t nc1 = nc/C0, c = nc % C0;
        const float* inpdata = (const float*)inp.data() + nc1*(H*W*C0) + c;
        float* outdata = (float*)out.data() + nc1*C0 + c;
        double s = 0;
        for (int64_t xy = 0; xy < H*W; xy++) {
            s += inpdata[xy*C0];
        }
        *outdata = (float)(s*scale);
    }
}

void test_globavgpool()
{
    int64_t N = 2, C1 = 3, C0 = 8, H = 3, W = 4;
    printf("=========== GLOBAL AVERAGE POOLING TEST ===========\n");
    Tensor a({{N, C1, H, W, C0}, LAYOUT_NCHWc}, CV_32F);
    std::vector<Tensor> c0;
    Tensor t_ref;
    std::vector<Buffer> tmp;
    float* adata = a.ptr<float>();

    for (int64_t i = 0; i < N*C1*H*W*C0; i++) {
        adata[i] = (float)::sin((double)i);
    }
    Net2 net;
    Graph g = net.newGraph("main", {}, {});
    Op op = GlobalAveragePoolOp::create();

    std::cout << "input: ";
    a.dump(std::cout, 0) << "\n";
    std::cout << "==============================\n";

    op->forward(net, g, {a}, c0, tmp);
    std::cout << "result (shape = ";
    c0[0].sizetype().dump(std::cout) << "): ";
    c0[0].dump(std::cout, 0) << "\n";

    t_ref.fitSameDevice(c0[0], c0[0].size(), c0[0].type());
    ref_globavgpool(a, t_ref);

    Mat m0 = c0[0].getMat();
    Mat m1 = t_ref.getMat();
    printf("err = %.5g\n", norm(m0, m1, NORM_INF));
}

}}
