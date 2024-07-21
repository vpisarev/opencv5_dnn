// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/net2_impl.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv { namespace dnn {

static void global_average_pool_32f(const void* inp_, const TensorSize& size, void* out_)
{
    CV_Assert(size.layout == LAYOUT_NCHWc);
    int ndims = size.ndims;
    int64_t N = size.size[0], C1 = size.size[1], C0_ = size.size[ndims-1];
    int nlanes_ = (int)VTraits<v_float32>::vlanes();
    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);
    int64_t planesize = 1;
    N *= C1;
    for (int i = 2; i < ndims-1; i++)
        planesize *= size.size[i];

    parallel_for_(Range(0, (int)N), [&](const Range& r) {
        v_float32 scale = vx_setall_f32(planesize > 0 ? (float)(1./planesize) : 0);
        int64_t n0 = r.start, n1 = r.end;
        int nlanes = nlanes_, C0 = (int)C0_;
        for (int64_t n = n0; n < n1; n++) {
            const float* inp = (const float*)inp_ + planesize*C0*n;
            float* out = (float*)out_ + C0*n;
            int64_t planesize_C0 = planesize*C0;
            // by computing sum in blocks we probably increase accuracy
            int64_t BLOCK_SIZE = 256*C0, blocksize = 0;
            if (nlanes == C0) {
                v_float32 s0 = vx_setzero_f32();
                for (int64_t i = 0; i < planesize_C0; i += blocksize, inp += blocksize) {
                    blocksize = std::min(planesize_C0 - i, BLOCK_SIZE);
                    v_float32 bs0 = vx_setzero_f32();

                    for (int64_t j = 0; j < blocksize; j += C0)
                        bs0 = v_add(bs0, vx_load(inp + j));
                    s0 = v_add(s0, bs0);
                }
                s0 = v_mul(s0, scale);
                v_store(out, s0);
            } else if (nlanes*2 == C0) {
                v_float32 s0 = vx_setzero_f32(), s1 = s0;
                for (int64_t i = 0; i < planesize_C0; i += blocksize, inp += blocksize) {
                    blocksize = std::min(planesize_C0 - i, BLOCK_SIZE);
                    v_float32 bs0 = vx_setzero_f32(), bs1 = bs0;

                    for (int64_t j = 0; j < blocksize; j += C0) {
                        bs0 = v_add(bs0, vx_load(inp + j));
                        bs1 = v_add(bs1, vx_load(inp + j + nlanes));
                    }

                    s0 = v_add(s0, bs0);
                    s1 = v_add(s1, bs1);
                }
                s0 = v_mul(s0, scale);
                s1 = v_mul(s1, scale);
                v_store(out, s0);
                v_store(out + nlanes, s1);
            } else {
                memset(out, 0, C0*sizeof(out[0]));
                for (int64_t i = 0; i < planesize_C0; i += blocksize, inp += blocksize) {
                    blocksize = std::min(planesize_C0 - i, BLOCK_SIZE);
                    for (int c = 0; c < C0; c += nlanes*4) {
                        v_float32 s0 = vx_load(out + c);
                        v_float32 s1 = vx_load(out + c + nlanes);
                        v_float32 s2 = vx_load(out + c + nlanes*2);
                        v_float32 s3 = vx_load(out + c + nlanes*3);

                        v_float32 bs0 = vx_setzero_f32(), bs1 = bs0, bs2 = bs0, bs3 = bs0;

                        for (int64_t j = 0; j < blocksize; j += C0) {
                            bs0 = v_add(bs0, vx_load(inp + c + j));
                            bs1 = v_add(bs1, vx_load(inp + c + nlanes + j));
                            bs2 = v_add(bs2, vx_load(inp + c + nlanes*2 + j));
                            bs3 = v_add(bs3, vx_load(inp + c + nlanes*3 + j));
                        }

                        s0 = v_add(s0, bs0);
                        s1 = v_add(s1, bs1);
                        s2 = v_add(s2, bs2);
                        s3 = v_add(s3, bs3);
                        vx_store(out + c, s0);
                        vx_store(out + c + nlanes, s1);
                        vx_store(out + c + nlanes*2, s2);
                        vx_store(out + c + nlanes*3, s3);
                    }
                }
                for (int c = 0; c < C0; c += nlanes*2) {
                    v_float32 s0 = vx_load(out + c);
                    v_float32 s1 = vx_load(out + c + nlanes);
                    s0 = v_mul(s0, scale);
                    s1 = v_mul(s1, scale);
                    vx_store(out + c, s0);
                    vx_store(out + c + nlanes, s1);
                }
            }
        }
    });
}

template<typename _Tp>
void global_average_pool_16(const _Tp* inp_, const TensorSize& size, _Tp* out_)
{
    CV_Assert(size.layout == LAYOUT_NCHWc);
    int ndims = size.ndims;
    int64_t N = size.size[0], C1 = size.size[1], C0_ = size.size[ndims-1], C = size.C;
    int nlanes_ = (int)VTraits<v_float32>::vlanes();
    CV_Assert(C == C1*C0_);
    CV_Assert(C0_ == nlanes_ || C0_ == nlanes_*2 || C0_ % (nlanes_*4) == 0);
    int64_t planesize = 1;
    N *= C1;
    for (int i = 2; i < ndims-1; i++)
        planesize *= size.size[i];

    parallel_for_(Range(0, (int)N), [&](const Range& r) {
        v_float32 scale = vx_setall_f32(planesize > 0 ? (float)(1./planesize) : 0);
        int64_t n0 = r.start, n1 = r.end;
        int nlanes = nlanes_, C0 = (int)C0_;
        AutoBuffer<float> sbuf_(C0);
        float* sbuf = sbuf_.data();

        for (int64_t n = n0; n < n1; n++) {
            const _Tp* inp = inp_ + planesize*C0*n;
            _Tp* out = out_ + C0*n;
            int64_t planesize_C0 = planesize*C0;
            // by computing sum in blocks we probably increase accuracy
            int64_t BLOCK_SIZE = 256*C0, blocksize = 0;
            if (nlanes == C0) {
                v_float32 s0 = vx_setzero_f32();
                for (int64_t i = 0; i < planesize_C0; i += blocksize, inp += blocksize) {
                    blocksize = std::min(planesize_C0 - i, BLOCK_SIZE);
                    v_float32 bs0 = vx_setzero_f32();

                    for (int64_t j = 0; j < blocksize; j += C0)
                        bs0 = v_add(bs0, vx_load_expand(inp + j));
                    s0 = v_add(s0, bs0);
                }
                s0 = v_mul(s0, scale);
                v_pack_store(out, s0);
            } else if (nlanes*2 == C0) {
                v_float32 s0 = vx_setzero_f32(), s1 = s0;
                for (int64_t i = 0; i < planesize_C0; i += blocksize, inp += blocksize) {
                    blocksize = std::min(planesize_C0 - i, BLOCK_SIZE);
                    v_float32 bs0 = vx_setzero_f32(), bs1 = bs0;

                    for (int64_t j = 0; j < blocksize; j += C0) {
                        bs0 = v_add(bs0, vx_load_expand(inp + j));
                        bs1 = v_add(bs1, vx_load_expand(inp + j + nlanes));
                    }

                    s0 = v_add(s0, bs0);
                    s1 = v_add(s1, bs1);
                }
                s0 = v_mul(s0, scale);
                s1 = v_mul(s1, scale);
                v_pack_store(out, s0);
                v_pack_store(out + nlanes, s1);
            } else {
                memset(sbuf, 0, C0*sizeof(sbuf[0]));
                for (int64_t i = 0; i < planesize_C0; i += blocksize, inp += blocksize) {
                    blocksize = std::min(planesize_C0 - i, BLOCK_SIZE);
                    for (int c = 0; c < C0; c += nlanes*4) {
                        v_float32 s0 = vx_load(sbuf + c);
                        v_float32 s1 = vx_load(sbuf + c + nlanes);
                        v_float32 s2 = vx_load(sbuf + c + nlanes*2);
                        v_float32 s3 = vx_load(sbuf + c + nlanes*3);

                        v_float32 bs0 = vx_setzero_f32(), bs1 = bs0, bs2 = bs0, bs3 = bs0;

                        for (int64_t j = 0; j < blocksize; j += C0) {
                            bs0 = v_add(bs0, vx_load_expand(inp + c + j));
                            bs1 = v_add(bs1, vx_load_expand(inp + c + nlanes + j));
                            bs2 = v_add(bs2, vx_load_expand(inp + c + nlanes*2 + j));
                            bs3 = v_add(bs3, vx_load_expand(inp + c + nlanes*3 + j));
                        }

                        s0 = v_add(s0, bs0);
                        s1 = v_add(s1, bs1);
                        s2 = v_add(s2, bs2);
                        s3 = v_add(s3, bs3);
                        vx_store(sbuf + c, s0);
                        vx_store(sbuf + c + nlanes, s1);
                        vx_store(sbuf + c + nlanes*2, s2);
                        vx_store(sbuf + c + nlanes*3, s3);
                    }
                }
                for (int c = 0; c < C0; c += nlanes*2) {
                    v_float32 s0 = vx_load(sbuf + c);
                    v_float32 s1 = vx_load(sbuf + c + nlanes);
                    s0 = v_mul(s0, scale);
                    s1 = v_mul(s1, scale);
                    v_pack_store(out + c, s0);
                    v_pack_store(out + c + nlanes, s1);
                }
            }
        }
    });
}

static void global_average_pool_16f(const void* inp_, const TensorSize& size, void* out_)
{
    global_average_pool_16((const hfloat*)inp_, size, (hfloat*)out_);
}

static void global_average_pool_16bf(const void* inp_, const TensorSize& size, void* out_)
{
    global_average_pool_16((const bfloat*)inp_, size, (bfloat*)out_);
}

typedef void (*global_avgpool_func_t)(const void* inp, const TensorSize& size, void* out);

class GlobalAveragePoolOpImpl : public GlobalAveragePoolOp
{
public:
    GlobalAveragePoolOpImpl()
    {
    }
    virtual std::string_view name() const CV_OVERRIDE { return "GlobalAveragePool"; }
    virtual Op clone() const CV_OVERRIDE
    {
        return std::make_shared<GlobalAveragePoolOpImpl>();
    }

    virtual int minNumInputs() const CV_OVERRIDE { return 1; }
    virtual int maxNumInputs() const CV_OVERRIDE { return 1; }
    virtual int minNumOutputs() const CV_OVERRIDE { return 1; }
    virtual int maxNumOutputs() const CV_OVERRIDE { return 1; }

    int inferType(int inptype0) const
    {
        return inptype0;
    }

    virtual bool supportType(int, int depth) const CV_OVERRIDE
    {
        return depth == CV_32F || depth == CV_16F || depth == CV_16BF;
    }

    virtual bool alwaysSupportInplace() const CV_OVERRIDE
    {
        return false;
    }

    virtual int supportBlockLayout(int, int) const CV_OVERRIDE
    {
        return 1;
    }

    virtual int64_t getFLOPS(const std::vector<SizeType> &inputs,
                             const std::vector<SizeType> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        CV_Assert(outputs.size() == 1);
        // probably, there should be a coefficient in the case of complex reduction functions
        return (int64_t)inputs[0].size.total();
    }

    virtual void inferTypes(const Net2& net, const Graph& graph,
                            const std::vector<Arg>& inpargs,
                            const std::vector<int>& inptypes,
                            const std::vector<Arg>& outargs,
                            std::vector<int>& outtypes) const CV_OVERRIDE
    {
        int ninputs = (int)inpargs.size(), noutputs = (int)outargs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        CV_Assert((int)inptypes.size() == ninputs);
        CV_Assert(noutputs == 1);

        outtypes.resize(1);
        outtypes[0] = inferType(inptypes[0]);
    }

    TensorSize inferShapes_(const TensorSize& inpsize, int* actual_perm_=nullptr) const
    {
        int ndims = inpsize.ndims;
        TensorLayout inplayout = inpsize.layout;
        TensorSize outsize = inpsize;
        CV_Assert(inplayout == LAYOUT_NCHWc || inplayout == LAYOUT_NCHW);

        for (int i = 2; i < ndims - (inplayout == LAYOUT_NCHWc); i++)
            outsize.size[i] = 1;

        return outsize;
    }

    virtual void inferShapes(Net2& net, const Graph& graph,
                             const std::vector<Arg>& inpargs,
                             const std::vector<TensorSize>& inpshapes,
                             const std::vector<Arg>& outargs,
                             std::vector<TensorSize>& outshapes,
                             bool symbolic) const CV_OVERRIDE
    {
        int ninputs = (int)inpargs.size(), noutputs = (int)outargs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        CV_Assert(noutputs == 1);
        outshapes.resize(1);

        const TensorSize& inpsize = inpshapes[0];
        outshapes[0] = inferShapes_(inpsize);
    }

    virtual void forward(Net2& net, Graph& graph,
                        const std::vector<Tensor>& inputs,
                        std::vector<Tensor>& outputs,
                        std::vector<Buffer>& tempbufs) CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        const Tensor& inp = inputs[0];
        CV_Assert(inp.isContinuous());

        int inptype = inp.type(), outtype = inferType(inptype);
        TensorSize inpsize = inp.size();
        TensorSize outsize = inferShapes_(inpsize);
        outputs.resize(1);
        Tensor& out = outputs[0];
        out.fitSameDevice(inp, outsize, outtype);
        CV_Assert(out.isContinuous());

        if (inp.empty())
            return;

        const char* inptr0 = (const char*)inp.data();
        char* outptr0 = (char*)out.data();

        global_avgpool_func_t func =
            inptype == CV_32F ? global_average_pool_32f :
            inptype == CV_16F ? global_average_pool_16f :
            inptype == CV_16BF ? global_average_pool_16bf : nullptr;

        CV_Assert(func != nullptr);
        func(inptr0, inpsize, outptr0);
    }
};

GlobalAveragePoolOp::~GlobalAveragePoolOp() {}

Op GlobalAveragePoolOp::create()
{
    return std::make_shared<GlobalAveragePoolOpImpl>();
}

Arg globalAveragePool(Graph& graph, std::string_view opname,
                      std::string_view outname, Arg input)
{
    Op op = GlobalAveragePoolOp::create();
    return graph->append(opname, op, outname, {input});
}

}}
