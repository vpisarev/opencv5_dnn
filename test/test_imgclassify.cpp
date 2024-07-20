// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_all.hpp"
#include <math.h>

namespace cv { namespace dnn {

using std::string;

void topK(const Mat& m, std::vector<float>& probs, std::vector<int>& labels, int K)
{
    CV_Assert(m.rows == 1);
    CV_Assert(m.type() == CV_32F);
    int i, N = (int)m.total();
    const float* data0 = m.ptr<float>();
    K = std::min(N, K);
    std::vector<std::pair<float, int> > pairs(N);
    for (i = 0; i < N; i++) {
        pairs[i] = std::make_pair(-data0[i], i);
    }
    std::partial_sort(pairs.begin(), pairs.begin()+K, pairs.end());
    probs.resize(K);
    labels.resize(K);
    for (i = 0; i < K; i++) {
        probs[i] = -pairs[i].first;
        labels[i] = pairs[i].second;
    }
}

static void blobFromImageWithParams_(const Mat& image, Mat& blob, const Image2BlobParams& pparams)
{
    int rows = image.rows, cols = image.cols, planesize = rows*cols;
    int sizes[] = {1, 3, rows, cols};
    blob.create(4, sizes, CV_32F);

    float mr = (float)pparams.mean[0];
    float mg = (float)pparams.mean[1];
    float mb = (float)pparams.mean[2];

    float sr = (float)pparams.scalefactor[0];
    float sg = (float)pparams.scalefactor[1];
    float sb = (float)pparams.scalefactor[2];

    float* out = blob.ptr<float>();

    for (int y = 0; y < rows; y++, out += cols) {
        const uint8_t* inp = image.ptr<uint8_t>(y);
        for (int x = 0; x < cols; x++) {
            float b = (inp[x*3] - mb)*sb, g = (inp[x*3+1] - mg)*sg, r = (inp[x*3+2] - mr)*sr;
            out[x] = r;
            out[x+planesize] = g;
            out[x+planesize*2] = b;
        }
    }
}

void test_imgclassify()
{
    const char* model_root = getenv("OPENCV5_DNN_MODEL_PATH");
    if (model_root == 0 || strlen(model_root) == 0) {
        model_root = ".";
    }
    string model_path = model_root + string("/resnet50-v1-12.onnx");
    OnnxReaderParams params;
    Net2 net = readNetFromONNX2(model_path, params);
    //net.dump();
    string image_path = model_root + string("/images/sqcat.png");
    Mat image = imread(image_path, 1), blob;
    ::cv::dnn::Image2BlobParams pparams;
    pparams.scalefactor = Scalar(1.f/(255*0.229f), 1.f/(255*0.224f), 1.f/(255*0.225f));
    pparams.size = Size(224, 224);
    pparams.mean = Scalar(103.939f, 116.779f, 123.68f);
    pparams.swapRB = false;
    pparams.ddepth = CV_32F;
    pparams.datalayout = DNN_LAYOUT_NCHW;
    pparams.paddingmode = DNN_PMODE_CROP_CENTER;
    blobFromImageWithParams(image, blob, pparams);
    std::vector<Mat> inputs = {blob}, outputs;
    net.setTracingMode(DNN_TRACE_ALL);
    net.forward(inputs, outputs);
    CV_Assert(outputs.size() == 1);
    const Mat& out = outputs[0];
    printf("output: type=%s, ndims=%d (%d x %d)\n", typeToString(out.type()).c_str(), out.dims, out.rows, out.cols);
    std::vector<float> probs;
    std::vector<int> labels;
    int K = 5;
    topK(out, probs, labels, K);
    K = (int)probs.size();
    for (int i = 0; i < K; i++) {
        printf("%d. label=%d, prob=%.2g\n", i+1, labels[i], probs[i]);
    }
}

}}

