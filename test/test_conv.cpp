// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_all.hpp"
#include <math.h>
#include "opencv2/core/hal/intrin.hpp"

namespace cv { namespace dnn {

/*static void ref_conv2d(const Tensor& inp_, Tensor& out_, const ConvParams& params,
                       const Tensor& weights_, const Tensor& bias_)
{
    int ngroups = params.ngroups;
    TensorSize inpsize = inp_.size(), outsize = out_.size(), wsize = weights_.size();
    TensorLayout layout = inpsize.layout;
    CV_Assert(layout == LAYOUT_NCHWc);
    int ndims = inpsize.ndims;
    int wdims = wsize.ndims;
    CV_Assert(ndims == 5 && outsize.ndims == ndims);
    CV_Assert(wdims == 4);
    int64_t N = inpsize.size[0], C1 = inpsize.size[1], C0 = inpsize.size[ndims-1];
    int64_t Hi = inpsize.size[2], Wi = inpsize.size[3];
    int64_t K = wsize.size[0], WC = wsize.size[1], Hk = wsize.size[2], Wk = wsize.size[3];
    int64_t H0 = outsize.size[2], W0 = outsize.size[3];
    int64_t C1g = C1/ngroups, K1 = K/C0, K1g = K1/ngroups;
    int64_t iplanesize = Hi*Wi*C0;
    int64_t planesize = H0*W0*C0;
    int64_t HkWk = Hk*Wk;
    int64_t strides[] = {1, 1}, dilations[] = {1, 1}, pads[] = {0, 0, 0, 0};

    CV_Assert(outsize.size[0] == N);
    CV_Assert(outsize.size[ndims-1] == C0);
    CV_Assert(outsize.size[1]*outsize.size[ndims-1] == K);
    CV_Assert(outsize.size[1] == K1);
    CV_Assert(C1 % ngroups == 0);
    CV_Assert(K1 % ngroups == 0);

    if (!params.strides.empty()) {
        strides[0] = params.strides[0];
        strides[1] = params.strides[1];
    }

    if (!params.dilations.empty()) {
        dilations[0] = params.dilations[0];
        dilations[1] = params.dilations[1];
    }

    if (!params.pads.empty()) {
        pads[0] = params.pads[0];
        pads[1] = params.pads[1];
        pads[2] = params.pads[2];
        pads[3] = params.pads[3];
    }

    parallel_for_(Range(0, (int)(N*K1)), [&](const Range& r) {
        int64_t nk0 = r.start, nk1 = r.end;
        const float* inptr0 = (const float*)inp_.data();
        float* outptr0 = (float*)out_.data();
        const float* bptr0 = bias_.empty() ? nullptr : (const float*)bias_.data();
        const float* wptr0 = (const float*)weights_.data();
        AutoBuffer<float> sbuf(C0);
        float* sptr = sbuf.data();

        for (int64_t nk = nk0; nk < nk1; nk++) {
            int64_t n = nk/K1;
            int64_t k1 = nk - n*K1;
            int64_t g = k1/K1g;
            float* outptr = outptr0 + nk*planesize;
            for (int64_t y0 = 0; y0 < H0; y0++) {
                for (int64_t x0 = 0; x0 < W0; x0++, outptr += C0) {
                    int64_t yi_ = y0*strides[0] - pads[0];
                    int64_t xi_ = x0*strides[1] - pads[1];
                    for (int64_t c0 = 0; c0 < C0; c0++)
                        sptr[c0] = 0.f;

                    for (int64_t ky = 0; ky < Hk; ky++) {
                        int64_t yi = yi_ + ky*dilations[0];
                        if ((uint64_t)yi >= (uint64_t)Hi) continue;

                        for (int64_t kx = 0; kx < Wk; kx++) {
                            int64_t xi = xi_ + kx*dilations[1];
                            if ((uint64_t)xi >= (uint64_t)Wi) continue;

                            const float* inptr = inptr0 + (((n*ngroups + g)*C1g*Hi + yi)*Wi + xi)*C0;
                            const float* wptr = wptr0 + ((k1*C1g*Hk+ky)*Wk + kx)*C0;
                            for (int64_t c1 = 0; c1 < C1g; c1++, inptr += planesize, wptr += Hk*Wk*C0) {
                                for (int64_t c0 = 0; c0 < C0; c0++) {
                                    float inpval = inptr[c0];
                                    float w = wptr[c0*HkWk];
                                    sptr[c0] += inpval*w;
                                }
                            }
                        }
                    }

                    if (bptr0) {
                        for (int64_t c0 = 0; c0 < C0; c0++)
                            outptr[c0] = sptr[c0] + bptr0[k1*C0 + c0];
                    } else {
                        for (int64_t c0 = 0; c0 < C0; c0++)
                            outptr[c0] = sptr[c0];
                    }
                }
            }
        }
    });
}*/

void ref_conv2d(const Tensor& inp_, Tensor& out_, const ConvParams& params,
                const Tensor& weights_, const Tensor& bias_)
{
    int ngroups = params.ngroups;
    TensorSize inpsize = inp_.size(), outsize = out_.size(), wsize = weights_.size();
    TensorLayout layout = inpsize.layout;
    CV_Assert(layout == LAYOUT_NCHW);
    int ndims = inpsize.ndims;
    int wdims = wsize.ndims;
    CV_Assert(ndims == 4 && outsize.ndims == ndims);
    CV_Assert(wdims == 4);
    int64_t N = inpsize.size[0], C = inpsize.size[1];
    int64_t Hi = inpsize.size[2], Wi = inpsize.size[3];
    int64_t K = wsize.size[0], WCg = wsize.size[1], Hk = wsize.size[2], Wk = wsize.size[3];
    int64_t H0 = outsize.size[2], W0 = outsize.size[3];
    int64_t Cg = C/ngroups, Kg = K/ngroups;
    int64_t iplanesize = Hi*Wi;
    int64_t planesize = H0*W0;
    int64_t HkWk = Hk*Wk;
    int64_t strides[] = {1, 1}, dilations[] = {1, 1}, pads[] = {0, 0, 0, 0};

    CV_Assert(outsize.size[0] == N);
    CV_Assert(outsize.size[1] == K);
    CV_Assert(C % ngroups == 0);
    CV_Assert(K % ngroups == 0);

    if (!params.strides.empty()) {
        strides[0] = params.strides[0];
        strides[1] = params.strides[1];
    }

    if (!params.dilations.empty()) {
        dilations[0] = params.dilations[0];
        dilations[1] = params.dilations[1];
    }

    if (!params.pads.empty()) {
        pads[0] = params.pads[0];
        pads[1] = params.pads[1];
        pads[2] = params.pads[2];
        pads[3] = params.pads[3];
    }

    parallel_for_(Range(0, (int)(N*K)), [&](const Range& r) {
        int64_t nk0 = r.start, nk1 = r.end;
        const float* inp = inp_.ptr<float>();
        float* out = out_.ptr<float>();
        const float* bias = bias_.empty() ? nullptr : bias_.ptr<float>();
        const float* weights = weights_.ptr<float>();

        for (int nk = r.start; nk < r.end; nk++) {
            int64_t n = nk / K;
            int64_t k = nk - n*K;
            int64_t g = k/Kg;
            for (int64_t y0 = 0; y0 < H0; y0++) {
                for (int64_t x0 = 0; x0 < W0; x0++) {
                    int64_t yi_ = y0*strides[0] - pads[0];
                    int64_t xi_ = x0*strides[1] - pads[1];
                    float s = 0.f;

                    for (int64_t ky = 0; ky < Hk; ky++) {
                        int64_t yi = yi_ + ky*dilations[0];
                        if (yi < 0 || yi >= Hi) continue;

                        for (int64_t kx = 0; kx < Wk; kx++) {
                            int64_t xi = xi_ + kx*dilations[1];
                            if (xi < 0 || xi >= Wi) continue;
                            for (int64_t c = 0; c < Cg; c++) {
                                size_t ofs = (((n*ngroups + g)*Cg + c)*Hi + yi)*Wi + xi;
                                float inpval = inp[ofs];
                                float w = weights[((k*Cg + c)*Hk+ky)*Wk + kx];
                                s += inpval*w;
                            }
                        }
                    }
                    out[((n*K + k)*H0 + y0)*W0 + x0] = s + bias[k];
                }
            }
        }
    });
}


void test_conv()
{
    int nlanes = VTraits<v_float32>::vlanes();
    std::vector<Buffer> tmp;
    int iter, maxiter = 100;
    srand(0x1234567);

    printf("=========== CONVOLUTION TEST ===========\n");
    for (iter = 0; iter < maxiter; iter++) {
        std::cout << ".";
        std::cout.flush();
        int64_t N = (rand() % 4) + 1;
        int64_t C1 = (rand() % 3) + 1;
        int64_t C0 = (rand() % 2) ? nlanes*2 : nlanes;
        int64_t K1 = (rand() % 3) + 1;
        int Hk = (rand() % 5) + 1;
        int Wk = (rand() % 5) + 1;
        int SY = (rand() % 2) + 1;
        int SX = (rand() % 2) + 1;
        int DY = (rand() % 2) + 1;
        int DX = (rand() % 2) + 1;
        if (Hk == 1)
            SY = DY = 1;
        if (Wk == 1)
            SX = DX = 1;
        int ngroups = rand() % 3 + 1;
        int pad_y0 = (rand() % 2) ? Hk/2 : 0;
        int pad_y1 = (rand() % 2) ? Hk/2 : 0;
        int pad_x0 = (rand() % 2) ? Wk/2 : 0;
        int pad_x1 = (rand() % 2) ? Wk/2 : 0;
        if (ngroups > 1) {
            C1 = ((C1 + ngroups - 1) / ngroups) * ngroups;
            K1 = ((K1 + ngroups - 1) / ngroups) * ngroups;
        }
        int64_t K = K1*C0, C = C1*C0;

        int64_t Hi = ((rand() % 21) + 1)*SY + (Hk-1)*DY + pad_y0 + pad_y1 + 1;
        int64_t Wi = ((rand() % 21) + 1)*SX + (Wk-1)*DX + pad_x0 + pad_x1 + 1;

        ConvParams params;
        params.ngroups = ngroups;
        params.ksizes = {(int)K, (int)(C/ngroups), Hk, Wk};
        params.strides = {SY, SX};
        params.dilations = {DY, DX};
        params.pads = {pad_y0, pad_x0, pad_y1, pad_x1};

        Tensor a0({{N, C, Hi, Wi}, LAYOUT_NCHW}, CV_32F), a, c1, c0, c0_;
        Tensor w({{K, C/ngroups, Hk, Wk}, LAYOUT_UNKNOWN}, CV_32F);
        Tensor bias(TensorSize({K}, LAYOUT_UNKNOWN), CV_32F);

        size_t i, a_total = a0.total(), w_total = w.total(), b_total = bias.total();
        float* a_data = a0.ptr<float>();
        float* w_data = w.ptr<float>();
        float* b_data = bias.ptr<float>();

        for (i = 0; i < a_total; i++) {
            float x = sinf((float)i*0.1f);
            a_data[i] = x*x;
        }

        for (i = 0; i < w_total; i++) {
            float x = cosf((float)i*0.1f);
            w_data[i] = x;
        }

        for (i = 0; i < b_total; i++) {
            float b = (float)(i + 1);
        }

        Net2 net;
        Graph graph = net.newGraph("main", {}, {});

        Op nchw2block = TransformLayoutOp::create(LAYOUT_NCHWc, C0);
        Op block2nchw = TransformLayoutOp::create(LAYOUT_NCHW);
        Op conv = ConvOp::create(params);
        dynamic_cast<ConvOp*>(conv.get())->setWeights(w, bias, C0);

        nchw2block->forward(net, graph, {a0}, a, tmp);
        conv->forward(net, graph, {a}, c0_, tmp);
        block2nchw->forward(net, graph, {c0_}, c0, tmp);

        TensorSize outsize = ref_conv_infer_shapes(a0.size(), params, w.size());
        c1.fitSameDevice(a, outsize, a.type());
        ref_conv2d(a0, c1, params, w, bias);

        Mat m0 = c0.getMat();
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
            a0.size().dump(std::cout) << "\n";
            std::cout << "output shape: ";
            c0.size().dump(std::cout) << "\n";
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
