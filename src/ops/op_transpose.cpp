// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/net2_impl.hpp"

namespace cv { namespace dnn {

TransposeOp::~TransposeOp() {}

template <typename _Tp>
void transpose(const _Tp* inp_, int64_t istep, int64_t istep0, int64_t istep1,
               _Tp* out_, int64_t ostep, int64_t ostep0, int64_t ostep1,
               int64_t npix, int64_t C0, int64_t C1, int64_t C)
{
    
    CV_Assert(C0 % 8 == 0 || C0 == 4 || C1 == 1);
    CV_Assert(istep0 == 1 || ostep0 == 1);
    const int64_t dC0 = std::min(C0, (int64_t)8);
    for (int64_t c1 = 0; c1 < C1; c1++) {
        for (int64_t c0 = 0; c0 < C0; c0 += dC0) {
            const _Tp* inp = inp_ + istep0*c0 + istep1*c1;
            _Tp* out = out_ + ostep0*c0 + ostep1*c1;
            int64_t dc = std::min(C - (c1*C0 + c0), dC0);
            if (dc == 8) {
                if (istep0 == 1) {
                    for (int64_t i = 0; i < npix; i++, inp += istep, out += ostep) {
                        _Tp x0 = inp[0], x1 = inp[1], x2 = inp[2], x3 = inp[3];
                        _Tp x4 = inp[4], x5 = inp[5], x6 = inp[6], x7 = inp[7];
                        out[0] = x0; out[ostep0] = x1; out[ostep0*2] = x2; out[ostep0*3] = x3;
                        out[ostep0*4] = x4; out[ostep0*5] = x5; out[ostep0*6] = x6; out[ostep0*7] = x7;
                    }
                } else {
                    for (int64_t i = 0; i < npix; i++, inp += istep, out += ostep) {
                        _Tp x0 = inp[0], x1 = inp[istep0], x2 = inp[istep0*2], x3 = inp[istep0*3];
                        _Tp x4 = inp[istep0*4], x5 = inp[istep0*5], x6 = inp[istep0*6], x7 = inp[istep0*7];
                        out[0] = x0; out[1] = x1; out[2] = x2; out[3] = x3;
                        out[4] = x4; out[5] = x5; out[6] = x6; out[7] = x7;
                    }
                }
            } else if (dc == 4) {
                if (istep0 == 1) {
                    for (int64_t i = 0; i < npix; i++, inp += istep, out += ostep) {
                        _Tp x0 = inp[0], x1 = inp[1], x2 = inp[2], x3 = inp[3];
                        out[0] = x0; out[ostep0] = x1; out[ostep0*2] = x2; out[ostep0*3] = x3;
                    }
                } else {
                    for (int64_t i = 0; i < npix; i++, inp += istep, out += ostep) {
                        _Tp x0 = inp[0], x1 = inp[istep0], x2 = inp[istep0*2], x3 = inp[istep0*3];
                        out[0] = x0; out[1] = x1; out[2] = x2; out[3] = x3;
                    }
                }
            } else if (dc == 3 && ostep0 == 1 && ostep == C0) {
                memset(out, 0, npix*C0*sizeof(out[0]));
                for (int64_t i = 0; i < npix; i++, inp += istep, out += ostep) {
                    _Tp x0 = inp[0], x1 = inp[istep0], x2 = inp[istep0*2];
                    out[0] = x0; out[1] = x1; out[2] = x2;
                }
            } else {
                for (int64_t i = 0; i < npix; i++, inp += istep, out += ostep) {
                    int64_t c = 0;
                    for (; c < dc; c++)
                        out[ostep0*c] = inp[istep0*c];
                    for (; c < dC0; c++)
                        out[ostep0*c] = 0;
                }
            }
        }
    }
}

#undef CV_TRANSPOSE_IMPL
#define CV_TRANSPOSE_IMPL(typ, suffix) \
static void transpose_##suffix(const void* inp_, int64_t istep, int64_t istep0, int64_t istep1, \
                               void* out_, int64_t ostep, int64_t ostep0, int64_t ostep1, \
                               int64_t npix, int64_t C0, int64_t C1, int64_t C) \
{ \
    transpose((const typ*)inp_, istep, istep0, istep1, \
              (typ*)out_, ostep, ostep0, ostep1, npix, C0, C1, C); \
}

CV_TRANSPOSE_IMPL(uint8_t, 8u)
CV_TRANSPOSE_IMPL(uint16_t, 16u)
CV_TRANSPOSE_IMPL(uint32_t, 32u)
CV_TRANSPOSE_IMPL(uint64_t, 64u)

typedef void (*transpose_func_t)(const void* inp, int64_t istep, int64_t istep0, int64_t istep1,
                                        void* out, int64_t ostep, int64_t ostep0, int64_t ostep1,
                                        int64_t npix, int64_t C0, int64_t C1, int64_t C);

class TransposeOpImpl : public TransposeOp
{
public:
    TransposeOpImpl(const std::vector<int>& perm_)
    {
        perm = perm_;
    }
    virtual std::string_view name() const CV_OVERRIDE { return "Transpose"; }
    virtual Op clone() const CV_OVERRIDE
    {
        return std::make_shared<TransposeOpImpl>(perm);
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
        size_t esz = CV_ELEM_SIZE(depth);
        return esz == 1 || esz == 2 || esz == 4 || esz == 8;
    }

    virtual bool alwaysSupportInplace() const CV_OVERRIDE
    {
        return false;
    }

    virtual int64_t getFLOPS(const std::vector<SizeType> &inputs,
                             const std::vector<SizeType> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        CV_Assert(outputs.size() == 1);
        // probably, there should be a coefficient in the case of complex reduction functions
        return (int64_t)std::max(inputs[0].size.total(), outputs[0].size.total());
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
        int actual_perm_buf[TensorSize::MAX_DIMS];
        int* actual_perm = actual_perm_ ? actual_perm_ : actual_perm_buf;
        bool permmask[TensorSize::MAX_DIMS];
        int ndims = inpsize.ndims;
        TensorLayout inplayout = inpsize.layout;
        TensorSize outsize = inpsize;
        CV_Assert(inplayout != LAYOUT_NCHWc);

        bool NHWC2NCHW = true, NCHW2NHWC = true;

        if (perm.empty()) {
            for (int i = 0; i < ndims; i++)
                actual_perm[i] = ndims-i-1;
            NHWC2NCHW = NCHW2NHWC = false;
        } else {
            CV_Assert(perm.size() == (size_t)ndims);
            for (int i = 0; i < ndims; i++)
                permmask[i] = false;
            for (int i = 0; i < ndims; i++) {
                int p = perm[i];
                if (p < 0 || p >= ndims) {
                    CV_Error(Error::StsOutOfRange, "some axes in Transpose are out of range");
                }
                if (permmask[p]) {
                    CV_Error(Error::StsOutOfRange, "there are duplicates in the permutation vector");
                }
                // NHWC2NCHW: {0, 3, 1, 2}
                // NCHW2NHWC: {0, 2, 3, 1}
                if (i == 0 && p != 0)
                    NHWC2NCHW = NCHW2NHWC = false;
                if ((i == 1 && p != ndims-1) || (i > 1 && p != i-1))
                    NHWC2NCHW = false;
                if ((i == ndims-1 && p != 1) || (i > 0 && i < ndims-1 && p != i+1))
                    NCHW2NHWC = false;
                permmask[p] = true;
                actual_perm[i] = p;
            }
        }

        outsize.layout =
            inplayout == LAYOUT_NHWC && NHWC2NCHW ? LAYOUT_NCHW :
            inplayout == LAYOUT_NCHW && NCHW2NHWC ? LAYOUT_NHWC : LAYOUT_UNKNOWN;

        for (int i = 0; i < ndims; i++)
            outsize.size[i] = inpsize.size[actual_perm[i]];

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

        size_t esz = CV_ELEM_SIZE(inptype);
        const char* inptr0 = (const char*)inp.data();
        char* outptr0 = (char*)out.data();

        transpose_func_t transpose_func =
            //esz == 1 ? transpose_8u :
            //esz == 2 ? transpose_16u :
            //esz == 4 ? transpose_32u :
            //esz == 8 ? transpose_64u :
            nullptr;

        CV_Assert(transpose_func != nullptr);
#if 0
        parallel_for_(Range(0, (int)nblocks), [&](const Range& r) {
            int64_t start = r.start*allplanes/nblocks;
            int64_t end = r.end*allplanes/nblocks;
            int64_t npix = 0;

            for (int64_t ofs = start; ofs < end; ofs += npix) {
                int64_t sample_idx = ofs/planesize;
                int64_t rawofs = ofs - sample_idx*planesize;
                npix = std::min(planesize - rawofs, end - ofs);
                const char* inptr = inptr0 + (inplanesizeC*sample_idx + istep*rawofs)*esz;
                char* outptr = outptr0 + (outplanesizeC*sample_idx + ostep*rawofs)*esz;
                transpose_func(inptr, istep, istep0, istep1,
                                      outptr, ostep, ostep0, ostep1,
                                      npix, C0_, C1_, C);
            }
        });
#endif
    }
};

Op TransposeOp::create(const std::vector<int>& perm)
{
    return std::make_shared<TransposeOpImpl>(perm);
}

Arg transpose(Graph& graph, std::string_view opname, std::string_view outname,
              Arg input, const std::vector<int>& perm)
{
    Op op = TransposeOp::create(perm);
    return graph->append(opname, op, outname, {input});
}

}}

