// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"

namespace cv { namespace dnn {

TransformLayoutOp::~TransformLayoutOp() {}

template <typename _Tp>
void transform_layout(const _Tp* inp_, int64_t istep, int64_t istep0, int64_t istep1,
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

#undef CV_TRANSFORM_LAYOUT_IMPL
#define CV_TRANSFORM_LAYOUT_IMPL(typ, suffix) \
static void transform_layout_##suffix(const void* inp_, int64_t istep, int64_t istep0, int64_t istep1, \
                                      void* out_, int64_t ostep, int64_t ostep0, int64_t ostep1, \
                                      int64_t npix, int64_t C0, int64_t C1, int64_t C) \
{ \
    transform_layout((const typ*)inp_, istep, istep0, istep1, \
                     (typ*)out_, ostep, ostep0, ostep1, npix, C0, C1, C); \
}

CV_TRANSFORM_LAYOUT_IMPL(uint8_t, 8u)
CV_TRANSFORM_LAYOUT_IMPL(uint16_t, 16u)
CV_TRANSFORM_LAYOUT_IMPL(uint32_t, 32u)
CV_TRANSFORM_LAYOUT_IMPL(uint64_t, 64u)

typedef void (*transform_layout_func_t)(const void* inp, int64_t istep, int64_t istep0, int64_t istep1,
                                        void* out, int64_t ostep, int64_t ostep0, int64_t ostep1,
                                        int64_t npix, int64_t C0, int64_t C1, int64_t C);

class TransformLayoutOpImpl : public TransformLayoutOp
{
public:
    TransformLayoutOpImpl(TensorLayout layout_, int64_t C0_)
    {
        layout = layout_;
        C0 = C0_;
    }
    virtual std::string_view name() const CV_OVERRIDE { return "TransformLayout"; }
    virtual Op clone() const CV_OVERRIDE
    {
        return std::make_shared<TransformLayoutOpImpl>(layout, C0);
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

    TensorSize inferShapes_(const TensorSize& inpsize) const
    {
        int ndims = inpsize.ndims;
        TensorLayout inplayout = inpsize.layout;
        CV_Assert(layout == LAYOUT_NCHWc || layout == LAYOUT_NCHW || layout == LAYOUT_NHWC);
        CV_Assert(inplayout == LAYOUT_NCHWc || inplayout == LAYOUT_NCHW || inplayout == LAYOUT_NHWC);

        if (layout == inplayout) {
            // identity
            CV_Assert(layout != LAYOUT_NCHWc || C0 == inpsize.size[ndims-1]);
            return inpsize;
        }

        // non-block => block
        if (layout == LAYOUT_NCHWc)
            return inpsize.toBlock(C0);

        // block => non-block
        if (inplayout == LAYOUT_NCHWc)
            return inpsize.fromBlock(layout);

        TensorSize outsize = inpsize;
        outsize.layout = layout;

        // NHWC => NCHW
        if (layout == LAYOUT_NCHW) {
            CV_Assert(inplayout == LAYOUT_NHWC);
            int64_t C = inpsize.size[ndims-1];
            for (int i = 2; i < ndims; i++)
                outsize.size[i] = inpsize.size[i-1];
            outsize.size[1] = C;
        } else {
            // NCHW => NHWC
            CV_Assert(layout == LAYOUT_NHWC && inplayout == LAYOUT_NCHW);
            int64_t C = inpsize.size[1];
            for (int i = 2; i < ndims; i++)
                outsize.size[i-1] = inpsize.size[i];
            outsize.size[ndims-1] = C;
        }
        return outsize;
    }

    virtual void inferShapes(const Net2& net, const Graph& graph,
                            const std::vector<Arg>& inpargs,
                            const std::vector<SizeType>& inpst,
                            const std::vector<Arg>& outargs,
                            std::vector<SizeType>& outst,
                            std::vector<size_t>& tempbufs) const CV_OVERRIDE
    {
        int ninputs = (int)inpargs.size(), noutputs = (int)outargs.size();
        CV_Assert(minNumInputs() <= ninputs && ninputs <= maxNumInputs());
        CV_Assert(noutputs == 1);
        outst.resize(1);

        const TensorSize& inpsize = inpst[0].size;
        TensorSize& outsize = outst[0].size;

        outsize = inferShapes_(inpsize);
        outst[0].type = inferType(inpst[0].type);
        tempbufs.assign(1, (size_t)0);
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

        TensorLayout inplayout = inpsize.layout;
        TensorLayout outlayout = outsize.layout;

        if (inp.empty())
            return;

        if (inplayout == outlayout) {
            inp.copyTo(out);
            return;
        }

        int inp_ndims = inpsize.ndims;
        int out_ndims = outsize.ndims;
        int64_t N = inpsize.size[0];
        int64_t C = inplayout == LAYOUT_NCHWc ? inpsize.C :
            inpsize.size[inplayout == LAYOUT_NCHW ? 1 : inp_ndims-1];
        int64_t inptotal = (int64_t)inp.total();
        int64_t outtotal = (int64_t)out.total();
        int64_t inplanesizeC = inptotal / N;
        int64_t outplanesizeC = outtotal / N;
        int64_t planesize = (inplayout != LAYOUT_NCHWc ? inplanesizeC : outplanesizeC)/C;
        int64_t allplanes = planesize*N;

        const int64_t BLOCK_SIZE = 1 << 17;
        int64_t nblocks = (outtotal + BLOCK_SIZE - 1)/BLOCK_SIZE;
        nblocks = std::min(nblocks, allplanes);

        size_t esz = CV_ELEM_SIZE(inptype);
        int64_t istep0, istep1=0, istep;
        int64_t ostep0, ostep1=0, ostep;
        int64_t C0_ = C, C1_ = 1;

        if (inplayout == LAYOUT_NCHWc || outlayout == LAYOUT_NCHWc) {
            C0_ = inplayout == LAYOUT_NCHWc ? inpsize.size[inp_ndims-1] : outsize.size[out_ndims-1];
            C1_ = (C + C0_ - 1)/C0_;
        }

        if (inplayout == LAYOUT_NCHW) {
            istep = 1;
            istep0 = planesize;
        } else if (inplayout == LAYOUT_NHWC) {
            istep = C;
            istep0 = 1;
        } else {
            istep = C0_;
            istep0 = 1;
            istep1 = planesize*C0_;
        }

        if (outlayout == LAYOUT_NCHW) {
            ostep = 1;
            ostep0 = planesize;
        } else if (outlayout == LAYOUT_NHWC) {
            ostep = C;
            ostep0 = 1;
        } else {
            ostep = C0_;
            ostep0 = 1;
            ostep1 = planesize*C0_;
        }

        const char* inptr0 = (const char*)inp.data();
        char* outptr0 = (char*)out.data();

        transform_layout_func_t transform_layout_func =
            esz == 1 ? transform_layout_8u :
            esz == 2 ? transform_layout_16u :
            esz == 4 ? transform_layout_32u :
            esz == 8 ? transform_layout_64u : nullptr;

        CV_Assert(transform_layout_func != nullptr);

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
                transform_layout_func(inptr, istep, istep0, istep1,
                                      outptr, ostep, ostep0, ostep1,
                                      npix, C0_, C1_, C);
            }
        });
    }
};

Op TransformLayoutOp::create(TensorLayout layout, int64_t C0)
{
    return std::make_shared<TransformLayoutOpImpl>(layout, C0);
}

}}
