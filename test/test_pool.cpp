// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_all.hpp"
#include <math.h>
#include "opencv2/core/hal/intrin.hpp"

namespace cv { namespace dnn {

static void ref_globavgpool(const Tensor& inp_, Tensor& out_)
{
    TensorSize size = inp_.size();
    int64_t N = size.size[0], C1 = size.size[1], H = size.size[2], W = size.size[3], C0 = size.size[4];
    double scale = H*W > 0 ? 1./(H*W) : 0;
    for (int64_t nc = 0; nc < N*C1*C0; nc++) {
        int64_t nc1 = nc/C0, c = nc % C0;
        const float* inp = (const float*)inp_.data() + nc1*(H*W*C0) + c;
        float* out = (float*)out_.data() + nc1*C0 + c;
        double s = 0;
        for (int64_t xy = 0; xy < H*W; xy++) {
            s += inp[xy*C0];
        }
        *out = (float)(s*scale);
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

static void ref_maxpool(const Tensor& inp_, Tensor& out_, const ConvParams& params)
{
    TensorSize inpsize = inp_.size();
    TensorSize outsize = ref_conv_infer_shapes(inpsize, params);

    out_.fitSameDevice(inp_, outsize, inp_.type());
    CV_Assert(inp_.type() == CV_32F);

    int64_t N = outsize.size[0], C1 = outsize.size[1], C0 = outsize.size[4];

    parallel_for_(Range(0, (int)(N*C1*C0)), [&](const Range& r) {
        int64_t H = outsize.size[2], W = outsize.size[3];
        int64_t Hi = inpsize.size[2], Wi = inpsize.size[3];
        int64_t Hk = params.ksizes[0], Wk = params.ksizes[1];
        int64_t DY = params.dilations[0], DX = params.dilations[1];
        int64_t SY = params.strides[0], SX = params.strides[1];
        int64_t pad_y0 = params.pads[0], pad_x0 = params.pads[1];
        int64_t pad_y1 = params.pads[2], pad_x1 = params.pads[3];
        int nc0 = r.start, nc1 = r.end;

        for (int nc = nc0; nc < nc1; nc++) {
            int n = nc / C0;
            int c = nc % C0;
            const float* inp = (const float*)inp_.data() + n*Hi*Wi*C0 + c;
            float* out = (float*)out_.data() + n*H*W*C0 + c;

            for (int64_t y0 = 0; y0 < H; y0++) {
                int64_t yi_ = y0*SY - pad_y0;
                for (int64_t x0 = 0; x0 < W; x0++, out += C0) {
                    int64_t xi_ = x0*SX - pad_x0;
                    float s0 = -FLT_MAX;
                    for (int64_t ky = 0; ky < Hk; ky++) {
                        for (int64_t kx = 0; kx < Wk; kx++) {
                            int64_t yi = yi_ + ky*DY;
                            int64_t xi = xi_ + kx*DX;
                            float v0;
                            if ((uint64_t)yi >= (uint64_t)Hi || (uint64_t)xi >= (uint64_t)Wi)
                                continue;
                            v0 = inp[(yi*Wi + xi)*C0];
                            s0 = std::max(s0, v0);
                        }
                    }
                    *out = s0;
                }
            }
        }
    });
}

void test_maxpool()
{
    int nlanes = VTraits<v_float32>::vlanes();
    std::vector<Buffer> tmp;
    int iter, maxiter = 100;
    srand(0x1234567);

    printf("=========== MAX POOLING TEST ===========\n");
    for (iter = 0; iter < maxiter; iter++) {
        std::cout << ".";
        std::cout.flush();
        int64_t N = (rand() % 4) + 1;
        int64_t C1 = (rand() % 3) + 1;
        int64_t C0 = (rand() % 2) ? nlanes*2 : nlanes;
        int Hk = (rand() % 5) + 1;
        int Wk = (rand() % 5) + 1;
        int SY = (rand() % 2) + 1;
        int SX = (rand() % 2) + 1;
        int DY = (rand() % 2) + 1;
        int DX = (rand() % 2) + 1;
        int pad_y0 = (rand() % 2) ? Hk/2 : 0;
        int pad_y1 = (rand() % 2) ? Hk/2 : 0;
        int pad_x0 = (rand() % 2) ? Wk/2 : 0;
        int pad_x1 = (rand() % 2) ? Wk/2 : 0;

        int64_t Hi = ((rand() % 21) + 1)*SY + (Hk-1)*DY + pad_y0 + pad_y1 + 1;
        int64_t Wi = ((rand() % 21) + 1)*SX + (Wk-1)*DX + pad_x0 + pad_x1 + 1;

        ConvParams params;
        params.ksizes = {Hk, Wk};
        params.strides = {SY, SX};
        params.dilations = {DY, DX};
        params.pads = {pad_y0, pad_x0, pad_y1, pad_x1};

        Tensor a({{N, C1, Hi, Wi, C0}, LAYOUT_NCHWc}, CV_32F), c1;
        std::vector<Tensor> c0;

        size_t i, a_total = a.total();
        float* a_data = (float*)a.data();

        for (i = 0; i < a_total; i++) {
            float x = sinf((float)i*0.1f);
            a_data[i] = x*x;
        }

        Net2 net;
        Graph graph = net.newGraph("main", {}, {});

        Op op = MaxPoolOp::create(params, false, true);
        op->forward(net, graph, {a}, c0, tmp);

        ref_maxpool(a, c1, params);

        Mat m0 = c0[0].getMat();
        Mat m1 = c1.getMat();
        double err = norm(m0, m1, NORM_INF);
        if (err > 1e-2) {
            printf("\nFAILED at iter=%d!\n", iter);
            double minVal0, maxVal0, minVal1, maxVal1;
            minMaxIdx(m0, &minVal0, &maxVal0);
            minMaxIdx(m1, &minVal1, &maxVal1);
            printf("err = %.5g, minv=%.5g, maxv=%.5g, ref minv=%.5g, ref maxv=%.5g\n", err, minVal0, maxVal0, minVal1, maxVal1);
            std::cout << "params: ";
            params.dump(std::cout) << "\n";
            std::cout << "input shape: ";
            a.size().dump(std::cout) << "\n";
            std::cout << "output shape: ";
            c0[0].size().dump(std::cout) << "\n";
            std::cout << "ref output shape: ";
            c1.size().dump(std::cout) << "\n";
            break;
        }
    }

    if (iter == maxiter) {
        printf("\nOK\n");
    }
}

}}
