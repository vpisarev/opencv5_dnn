// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"
#include <math.h>

namespace cv { namespace dnn {

ReshapeOp::~ReshapeOp() {}

class ReshapeOpImpl : public ReshapeOp
{
public:
    ReshapeOpImpl(bool allowzero_)
    {
        allowZero = allowzero_;
    }
    virtual std::string_view name() const CV_OVERRIDE { return "Reshape"; }
    virtual Op clone() const CV_OVERRIDE
    {
        return std::make_shared<ReshapeOpImpl>(allowZero);
    }

    virtual int minNumInputs() const CV_OVERRIDE { return 2; }
    virtual int maxNumInputs() const CV_OVERRIDE { return 2; }
    virtual int minNumOutputs() const CV_OVERRIDE { return 1; }
    virtual int maxNumOutputs() const CV_OVERRIDE { return 1; }

    int inferType(int inptype0) const
    {
        return inptype0;
    }

    virtual bool supportType(int, int) const CV_OVERRIDE
    {
        return true;
    }

    virtual bool alwaysSupportInplace() const CV_OVERRIDE
    {
        return true;
    }

    virtual int64_t getFLOPS(const std::vector<SizeType> &inputs,
                             const std::vector<SizeType> &outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        return (int64_t)inputs[0].size.total();
    }

    TensorSize inferShapes_(const TensorSize& inpsize, const Tensor& shape) const
    {
        CV_Assert(inpsize.layout != LAYOUT_NCHWc);

        size_t inptotal = inpsize.total();
        TensorSize outsize = inpsize;

        if (shape.empty()) {
            if (inptotal != 1) {
                CV_Error(Error::StsError, "");
            }
            outsize.ndims = 0;
            return outsize;
        }

        CV_Assert(shape.ndims() <= 1);
        int out_ndims = (int)shape.total();
        int inp_ndims = inpsize.ndims;

        CV_Assert(inp_ndims == out_ndims);

        int shapetype = shape.type();
        CV_Assert(shapetype == CV_32S || shapetype == CV_64S);

        const int32_t* shape32 = (const int32_t*)shape.data();
        const int64_t* shape64 = (const int64_t*)shape.data();
        int m1_idx = -1;
        size_t shapetotal = 1;

        for (int i = 0; i < inp_ndims; i++) {
            int64_t sz_i = shapetype == CV_32S ? (int64_t)shape32[i] : shape64[i];
            CV_Assert(sz_i >= -1);
            if (sz_i == -1) {
                if (m1_idx >= 0) {
                    CV_Error(Error::StsError, "more than one '-1' in Reshape's shape specification");
                }
                m1_idx = i;
                continue;
            }
            if (sz_i == 0)
                sz_i = allowZero ? sz_i : inpsize.size[i];
            outsize.size[i] = sz_i;
            shapetotal *= sz_i;
        }

        if (shapetotal == 0) {
            if (m1_idx >= 0) {
                CV_Error(Error::StsError, "when allowzero=true, shape spec cannot contain both '-1' and '0'");
            }
            if (inptotal != 0) {
                CV_Error(Error::StsError, "specified shape is empty while inptotal is not empty");
            }
        } else if (m1_idx >= 0) {
            outsize.size[m1_idx] = inptotal/shapetotal;
        }
        
        if (inptotal != outsize.total())
            CV_Error(Error::StsError, "the numbers of input elements and output elements don't match");

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
        const Tensor& axes = net.argTensor(inpargs[1]);
        outst[0].size = inferShapes_(inpsize, axes);
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
        TensorSize outsize = inferShapes_(inpsize, inputs[1]);
        outputs.resize(1);
        inp.reshape(outsize).copyTo(outputs[0]);
    }
};

Op ReshapeOp::create(bool allowzero)
{
    return std::make_shared<ReshapeOpImpl>(allowzero);
}

Arg reshape(Graph& graph, std::string_view opname,
            std::string_view outname, Arg input,
            Arg shape, bool allowzero)
{
    Op op = ReshapeOp::create(allowzero);
    return graph->append(opname, op, outname, {input, shape});
}

}}

