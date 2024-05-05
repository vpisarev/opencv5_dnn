// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#include <opencv2/dnn/layer_reg.private.hpp>
#include <opencv2/core/utils/fp_control_utils.hpp>
#include <opencv2/core/utils/logger.defines.hpp>
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#include <opencv2/core/utils/logger.hpp>

#include <algorithm>
#include <iostream>
#include <limits>
#include <set>
#include <string>

#include "../../engine/engine.hpp"
#include "opencv-onnx-pb-c.h"

namespace cv {
namespace dnn {

using std::vector;
using std::string;
using std::string_view;

static void onnxParseError(string_view ctx, string_view msg)
{
    throw std::runtime_error((!ctx.empty() ? " " + string(ctx) : "") + ": " + string(msg));
}

#define OnnxAssert(ctx, expr) if (!!(expr)) ; else onnxParseError(ctx, "assertion '" #expr "' is invalid")

static string onnxConcatCtx(string_view ctx, string_view subctx)
{
    return ctx.empty() ? string(subctx) : string(ctx) + ", " + string(subctx);
}

class OnnxImporter2
{
public:
    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;
    OnnxImporter2(Net2& net, const char* fileName);
    OnnxImporter2(Net2& net, const char* buffer, size_t bufsize);

protected:
    void init(Net2& net);
    bool parse(const char* fileName);
    bool parse(const char* buffer, size_t bufsize);
    Graph parseGraph(const OpenCVOnnx__GraphProto* proto, bool subgraph);
    typedef void (OnnxImporter2::*NodeParser)
        (string_view ctx, const OpenCVOnnx__NodeProto*,
         Graph& graph, const vector<Arg>&, const vector<Arg>&);

    Net2* net;
    typedef std::unordered_map<string, NodeParser> DispatchMap;
    typedef std::unordered_map<string, DispatchMap> DomainDispatchMap;
    DomainDispatchMap alldispatch;
    string filename;
    string defaultOnnxDomain = "ai.onnx";
    string frameworkName;
    std::set<string> unsupportedOps;

    void parseIf(string_view, const OpenCVOnnx__NodeProto*,
                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseLoop(string_view, const OpenCVOnnx__NodeProto*,
                   Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseScan(string_view, const OpenCVOnnx__NodeProto*,
                   Graph&, const vector<Arg>&, const vector<Arg>&);

    void parseBatchNormalization(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseCast(string_view, const OpenCVOnnx__NodeProto*,
                   Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseClip(string_view, const OpenCVOnnx__NodeProto*,
                   Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseConcat(string_view, const OpenCVOnnx__NodeProto*,
                     Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseConstant(string_view, const OpenCVOnnx__NodeProto*,
                       Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseConstantOfShape(string_view, const OpenCVOnnx__NodeProto*,
                              Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseConv(string_view, const OpenCVOnnx__NodeProto*,
                   Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseConvTranspose(string_view, const OpenCVOnnx__NodeProto*,
                            Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseCumSum(string_view, const OpenCVOnnx__NodeProto*,
                     Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseDepthToSpace(string_view, const OpenCVOnnx__NodeProto*,
                           Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseDetectionOutput(string_view, const OpenCVOnnx__NodeProto*,
                              Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseDropout(string_view, const OpenCVOnnx__NodeProto*,
                      Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseElemwiseBinary(string_view, const OpenCVOnnx__NodeProto*,
                             Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseElemwiseNary(string_view, const OpenCVOnnx__NodeProto*,
                           Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseElemwiseUnary(string_view, const OpenCVOnnx__NodeProto*,
                            Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseExpand(string_view, const OpenCVOnnx__NodeProto*,
                     Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseFlatten(string_view, const OpenCVOnnx__NodeProto*,
                      Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseGather(string_view, const OpenCVOnnx__NodeProto*,
                     Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseGemm(string_view, const OpenCVOnnx__NodeProto*,
                   Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseGlobalPool(string_view, const OpenCVOnnx__NodeProto*,
                         Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseGRU(string_view, const OpenCVOnnx__NodeProto*,
                  Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseImageScaler(string_view, const OpenCVOnnx__NodeProto*,
                          Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseInstanceNormalization(string_view, const OpenCVOnnx__NodeProto*,
                                    Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseLeakyRelu(string_view, const OpenCVOnnx__NodeProto*,
                        Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseLRN(string_view, const OpenCVOnnx__NodeProto*,
                  Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseLSTM(string_view, const OpenCVOnnx__NodeProto*,
                   Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseMatMul(string_view, const OpenCVOnnx__NodeProto*,
                     Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseMaxPool(string_view, const OpenCVOnnx__NodeProto*,
                      Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseMaxUnpool(string_view, const OpenCVOnnx__NodeProto*,
                        Graph&, const vector<Arg>&, const vector<Arg>&);
    void parsePad(string_view, const OpenCVOnnx__NodeProto*,
                  Graph&, const vector<Arg>&, const vector<Arg>&);
    void parsePooling(string_view, const OpenCVOnnx__NodeProto*,
                      Graph&, const vector<Arg>&, const vector<Arg>&);
    void parsePRelu(string_view, const OpenCVOnnx__NodeProto*,
                    Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseReduce(string_view, const OpenCVOnnx__NodeProto*,
                     Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseRelu(string_view, const OpenCVOnnx__NodeProto*,
                   Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseResize(string_view, const OpenCVOnnx__NodeProto*,
                     Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseReshape(string_view, const OpenCVOnnx__NodeProto*,
                      Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseShape(string_view, const OpenCVOnnx__NodeProto*,
                    Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseSlice(string_view, const OpenCVOnnx__NodeProto*,
                    Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseSoftMax(string_view, const OpenCVOnnx__NodeProto*,
                      Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseSplit(string_view, const OpenCVOnnx__NodeProto*,
                    Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseSqueeze(string_view, const OpenCVOnnx__NodeProto*,
                      Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseTranspose(string_view, const OpenCVOnnx__NodeProto*,
                        Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseUnsqueeze(string_view, const OpenCVOnnx__NodeProto*,
                        Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseUpsample(string_view, const OpenCVOnnx__NodeProto*,
                       Graph&, const vector<Arg>&, const vector<Arg>&);

    // Domain: com.microsoft
    // URL: https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md
    /*void parseDequantizeLinear(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQLinearAveragePool(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQLinearConcat(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQLinearConv(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQLinearElemwiseBinary(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQLinearGlobalAveragePool(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQLinearLeakyRelu(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQLinearMatMul(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQLinearSigmoid(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);
    void parseQuantizeLinear(string_view, const OpenCVOnnx__NodeProto*,
                                 Graph&, const vector<Arg>&, const vector<Arg>&);*/
};

struct OnnxTensor
{
    string name;
    Tensor t;
};

static int onnxDatatypeToDepth(string_view ctx, int datatype)
{
    int typ =
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__UNDEFINED ? -1 :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__UINT8 ? CV_8U :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__INT8 ? CV_8S :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__UINT16 ? CV_16U :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__INT16 ? CV_16S :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__UINT32 ? CV_32U :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__INT32 ? CV_32S :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__UINT64 ? CV_64U :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__INT64 ? CV_64S :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__BOOL ? CV_Bool :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT ? CV_32F :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__DOUBLE ? CV_64F :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16 ? CV_16F :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__BFLOAT16 ? CV_16BF :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX64 ? CV_32FC2 :
        datatype == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__COMPLEX128 ? CV_64FC2 : -2;
    if (typ < -1)
        onnxParseError(ctx, format("unsupported data_type %d", datatype));
    return typ;
}

static string onnxAttrTypeToString(OpenCVOnnx__AttributeProto__AttributeType tag)
{
    return tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__UNDEFINED ? "UNDEFINED" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT ? "FLOAT" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT ? "INT" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRING ? "STRING" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR ? "TENSOR" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH ? "GRAPH" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSOR ? "SPARSE_TENSOR" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TYPE_PROTO ? "TYPE_PROTO" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS ? "FLOATS" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS ? "INTS" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRINGS ? "STRINGS" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSORS ? "TENSORS" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPHS ? "GRAPHS" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__SPARSE_TENSORS ? "SPARSE_TENSORS" :
        tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TYPE_PROTOS ? "TYPE_PROTOS" :
        format("??? (tag=%d)", (int)tag);
}

template<typename fromT, typename toT>
static void onnxParseArray(Net2* net, string_view ctx,
                           fromT** arr, size_t nelems, vector<toT>& result);

template<typename fromT, typename toT> struct OnnxParseElem
{
    OnnxParseElem(Net2*, string_view) {}
    toT parse(const fromT* proto) const { return static_cast<toT>(*proto); }
};

template<> struct OnnxParseElem<OpenCVOnnx__OperatorSetIdProto, OnnxOpSet>
{
    OnnxParseElem(Net2*, string_view) {}
    OnnxOpSet parse(const OpenCVOnnx__OperatorSetIdProto* proto) {
        return std::make_pair(proto->version, proto->domain);
    }
};

template<> struct OnnxParseElem<OpenCVOnnx__TensorShapeProto__Dimension, OnnxTensorDim>
{
    OnnxParseElem(Net2*, string_view ctx_) : ctx(ctx_), idx(-1) {}
    OnnxTensorDim parse(const OpenCVOnnx__TensorShapeProto__Dimension* proto) {
        idx++;
        if (proto->value_case == OPENCV_ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM)
            return OnnxTensorDim(proto->dim_param);
        if (proto->value_case != OPENCV_ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE)
            onnxParseError(ctx, format("unknown type of dimension #%d", idx));
        return OnnxTensorDim(proto->dim_value);
    }
    string ctx;
    int idx;
};

template<> struct OnnxParseElem<OpenCVOnnx__ValueInfoProto, OnnxArgInfo>
{
    OnnxParseElem(Net2* net_, string_view ctx_) : net(net_), ctx(ctx_) {}
    OnnxArgInfo parse(const OpenCVOnnx__ValueInfoProto* proto) {
        OnnxArgInfo arginfo;
        arginfo.name = proto->name;
        string subctx = onnxConcatCtx(ctx,
                format("parsing value info '%s'", proto->name));
        if (proto->type->value_case == OPENCV_ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE) {
            arginfo.type = onnxDatatypeToDepth(subctx, proto->type->tensor_type->elem_type);
            const OpenCVOnnx__TensorShapeProto* shape = proto->type->tensor_type->shape;
            if (shape)
                onnxParseArray(net, subctx, shape->dim, shape->n_dim, arginfo.size);
        } else if (proto->type->value_case == OPENCV_ONNX__TYPE_PROTO__VALUE_SEQUENCE_TYPE) {
            onnxParseError(subctx, "sequences are not supported");
        } else {
            onnxParseError(subctx, format("unsupported value info tag %d",
                           (int)proto->type->value_case));
        }
        return arginfo;
    }
    Net2* net;
    string ctx;
};

static int64_t unpack_int64(uint8_t* p)
{
    uint64_t x = p[0] | ((uint64_t)p[1]<<8) | ((uint64_t)p[2]<<16) | ((uint64_t)p[3]<<24) |
        ((uint64_t)p[4]<<32) | ((uint64_t)p[5]<<40) | ((uint64_t)p[6]<<48) | ((uint64_t)p[7]<<56);
    return (int64_t)(x);
}

static Tensor onnxParseTensor(string_view ctx,
                              const OpenCVOnnx__TensorProto* tensor_proto)
{
    Tensor tensor;
    int i, n_dims = (int)tensor_proto->n_dims;
    size_t total = 1, elemsize;
    TensorSize shape;
    uchar* data;
    shape.ndims = n_dims;

    for (i = 0; i < n_dims; i++) {
        int64_t size_i = (int64_t)tensor_proto->dims[i];
        shape.size[i] = size_i;
        total *= size_i;
    }
    shape.layout = n_dims <= 2 ? LAYOUT_ND : LAYOUT_NCHW;
    int typ = onnxDatatypeToDepth(ctx, tensor_proto->data_type);
    if (typ < 0)
        onnxParseError(ctx, format("type of tensor '%s' is invalid (=%d)",
                        tensor_proto->name, tensor_proto->data_type));
    elemsize = CV_ELEM_SIZE(typ);
    tensor.fit(shape, typ);
    data = (uchar*)tensor.data();

    if (elemsize == 1 && tensor_proto->raw_data.len == total) {
        memcpy(data, tensor_proto->raw_data.data, total*elemsize);
    } else if (elemsize == 1 && tensor_proto->n_int32_data == total) {
        for(int j = 0; j < total; j++)
            data[j] = (uchar)(tensor_proto->int32_data[j]);
    } else if (elemsize == 4 && tensor_proto->n_float_data == total) {
        memcpy(data, tensor_proto->float_data, total*elemsize);
    } else if (elemsize == 4 &&
               tensor_proto->data_type == OPENCV_ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT16 &&
               tensor_proto->n_int32_data == total) {
        float* dst = (float*)data;
        for(size_t j = 0; j < total; j++) {
            ushort bits = (ushort)(tensor_proto->int32_data[j]);
            dst[j] = (float)float16_t::fromBits(bits);
        }
    } else if (elemsize == 4 && tensor_proto->raw_data.len == total*4) {
        uint32_t* dst = (uint32_t*)data;
        for(size_t j = 0; j < total; j++) {
            uint8_t* p = tensor_proto->raw_data.data + j*4;
            dst[j] = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
            ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        }
    } else if (elemsize == 4 && tensor_proto->n_int64_data == total) {
        int32_t* dst = (int32_t*)data;
        for(size_t j = 0; j < total; j++) {
            int64_t v = tensor_proto->int64_data[j];
            dst[j] = (int32_t)(v < INT_MIN ? INT_MIN : v > INT_MAX ? INT_MAX : (int)v);
        }
    } else if (elemsize == 4 && tensor_proto->raw_data.len == total*8) {
        int32_t* dst = (int32_t*)data;
        for(size_t j = 0; j < total; j++) {
            uint8_t* p = tensor_proto->raw_data.data + j*8;
            int64_t v = unpack_int64(p);
            dst[j] = (int32_t)(v < INT_MIN ? INT_MIN : v > INT_MAX ? INT_MAX : (int)v);
        }
    } else {
        onnxParseError(ctx, format("unsupported tensor data_type %d", (int)tensor_proto->data_type));
    }
    return tensor;
}

template<> struct OnnxParseElem<OpenCVOnnx__TensorProto, OnnxTensor>
{
    OnnxParseElem(Net2* net_, string_view ctx_) : net(net_), ctx(ctx_) {}
    OnnxTensor parse(const OpenCVOnnx__TensorProto* proto) {
        OnnxTensor tensor;
        tensor.name = proto->name;
        tensor.t = onnxParseTensor(ctx, proto);
        return tensor;
    }
    Net2* net;
    string ctx;
};

template<typename fromT, typename toT>
static void onnxParseArray(Net2* net, string_view ctx,
                           fromT** arr, size_t nelems, vector<toT>& result)
{
    OnnxParseElem<fromT, toT> elemParser(net, ctx);
    result.reserve(nelems);
    for (size_t i = 0; i < nelems; i++)
        result.push_back(elemParser.parse(arr[i]));
}


static int64_t onnxAttrInt(string_view ctx,
                            const OpenCVOnnx__NodeProto* node_proto,
                            string_view attr_name,
                            int64_t defval, bool* have_attr=nullptr)
{
    if (have_attr)
        *have_attr = false;
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name) {
            OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
            if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INT) {
                if (have_attr)
                    *have_attr = true;
                return (int64_t)attr_proto->i;
            } else {
                onnxParseError(ctx, format("unrecognized/unsupported type %s of attribute '%s' (must be INT)",
                                           onnxAttrTypeToString(tag).c_str(), attr_proto->name));
            }
        }
    }
    return defval;
}

static float onnxAttrFloat(string_view ctx,
                            const OpenCVOnnx__NodeProto* node_proto,
                            string_view attr_name,
                            float defval, bool* have_attr=nullptr)
{
    if (have_attr)
        *have_attr = false;
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name) {
            OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
            if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT) {
                if (have_attr)
                    *have_attr = true;
                return attr_proto->f;
            } else {
                onnxParseError(ctx, format("unrecognized/unsupported type %s of attribute '%s' (must be FLOAT)",
                                           onnxAttrTypeToString(tag).c_str(), attr_proto->name));
            }
        }
    }
    return defval;
}

static string onnxAttrString(string_view ctx,
                              const OpenCVOnnx__NodeProto* node_proto,
                              string_view attr_name,
                              string_view defval, bool* have_attr=nullptr)
{
    if (have_attr)
        *have_attr = false;
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name) {
            OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
            if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOAT) {
                if (have_attr)
                    *have_attr = true;
                return string((const char*)attr_proto->s.data, attr_proto->s.len);
            } else {
                onnxParseError(ctx, format("unrecognized/unsupported type %s of attribute '%s' (must be STRING)",
                                           onnxAttrTypeToString(tag).c_str(), attr_proto->name));
            }
        }
    }
    return string(defval);
}

static void onnxAttrInts(string_view ctx,
                          const OpenCVOnnx__NodeProto* node_proto,
                          string_view attr_name,
                          vector<int64_t>& values,
                          bool* have_attr=nullptr)
{
    if (have_attr)
        *have_attr = false;
    values.clear();
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name) {
            OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
            if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__INTS) {
                if (have_attr)
                    *have_attr = true;
                size_t i, n = attr_proto->n_ints;
                values.resize(n);
                for (i = 0; i < n; i++)
                    values[i] = (int64_t)attr_proto->ints[i];
                return;
            } else {
                onnxParseError(ctx, format("unrecognized/unsupported type %s of attribute '%s' (must be INTS)",
                                           onnxAttrTypeToString(tag).c_str(), attr_proto->name));
            }
        }
    }
}

static void onnxAttrFloats(string_view ctx,
                          const OpenCVOnnx__NodeProto* node_proto,
                          string_view attr_name,
                          vector<float>& values,
                          bool* have_attr=nullptr)
{
    if (have_attr)
        *have_attr = false;
    values.clear();
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name) {
            OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
            if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__FLOATS) {
                if (have_attr)
                    *have_attr = true;
                size_t i, n = attr_proto->n_floats;
                values.resize(n);
                for (i = 0; i < n; i++)
                    values[i] = attr_proto->floats[i];
                return;
            } else {
                onnxParseError(ctx, format("unrecognized/unsupported type %s of attribute '%s' (must be INTS)",
                                           onnxAttrTypeToString(tag).c_str(), attr_proto->name));
            }
        }
    }
}

static void onnxAttrStrings(string_view ctx,
                          const OpenCVOnnx__NodeProto* node_proto,
                          string_view attr_name,
                          vector<string>& values,
                          bool* have_attr=nullptr)
{
    if (have_attr)
        *have_attr = false;
    values.clear();
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name) {
            OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
            if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__STRINGS) {
                if (have_attr)
                    *have_attr = true;
                size_t i, n = attr_proto->n_strings;
                values.resize(n);
                for (i = 0; i < n; i++)
                    values[i] = string((const char*)attr_proto->strings[i].data,
                                       attr_proto->strings[i].len);
                return;
            } else {
                onnxParseError(ctx, format("unrecognized/unsupported type %s of attribute '%s' (must be INTS)",
                                           onnxAttrTypeToString(tag).c_str(), attr_proto->name));
            }
        }
    }
}

static const OpenCVOnnx__GraphProto* onnxAttrGraph(string_view ctx,
                          const OpenCVOnnx__NodeProto* node_proto,
                          string_view attr_name)
{
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name) {
            OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
            if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__GRAPH) {
                return attr_proto->g;
            } else {
                onnxParseError(ctx, format("unrecognized/unsupported type %s of attribute '%s' (must be GRAPH)",
                                           onnxAttrTypeToString(tag).c_str(), attr_proto->name));
            }
        }
    }
    return nullptr;
}

static Tensor onnxAttrTensor(string_view ctx,
                          const OpenCVOnnx__NodeProto* node_proto,
                          string_view attr_name)
{
    for (size_t j = 0; j < node_proto->n_attribute; j++) {
        OpenCVOnnx__AttributeProto* attr_proto = node_proto->attribute[j];
        if (attr_name == attr_proto->name) {
            OpenCVOnnx__AttributeProto__AttributeType tag = attr_proto->type;
            if (tag == OPENCV_ONNX__ATTRIBUTE_PROTO__ATTRIBUTE_TYPE__TENSOR) {
                return onnxParseTensor(ctx, attr_proto->t);
            } else {
                onnxParseError(ctx, format("unrecognized/unsupported type %s of attribute '%s' (must be TENSOR)",
                                           onnxAttrTypeToString(tag).c_str(), attr_proto->name));
            }
        }
    }
    return Tensor();
}


Graph OnnxImporter2::parseGraph(const OpenCVOnnx__GraphProto* proto, bool subgraph)
{
    string ctx = subgraph ? string() : "parsing subgraph '" + string(proto->name) + "'";
    vector<OnnxArgInfo> inputs, outputs, values;
    vector<OnnxTensor> initializers;
    vector<Arg> node_inputs, node_outputs;

    onnxParseArray(net, ctx, proto->input, proto->n_input, inputs);
    onnxParseArray(net, ctx, proto->output, proto->n_output, outputs);
    onnxParseArray(net, ctx, proto->value_info, proto->n_value_info, values);
    onnxParseArray(net, ctx, proto->initializer, proto->n_initializer, initializers);

    for (const OnnxTensor& t: initializers)
        net->newConstArg(t.name, t.t);

    for (int k = 0; k < 3; k++) {
        ArgKind argkind = k == 0 ? DNN_ARG_INPUT : k == 1 ? DNN_ARG_OUTPUT : DNN_ARG_TEMP;
        if (subgraph) argkind = DNN_ARG_TEMP;
        const vector<OnnxArgInfo>* graph_args = k == 0 ? &inputs : k == 1 ? &outputs : &values;
        for (const OnnxArgInfo& arginfo: *graph_args) {
            Arg arg = net->newArg(arginfo.name, argkind);
            if (k == 0)
                node_inputs.push_back(arg);
            else if (k == 1)
                node_outputs.push_back(arg);
        }
    }

    Graph graph = net->newGraph(proto->name, node_inputs);
    graph->setOutputs(node_outputs);

    for (size_t i = 0; i < proto->n_node; i++) {
        OpenCVOnnx__NodeProto* node_proto = proto->node[i];
        //printf("(%d/%d). parsing %s (%s)\n", (int)i, (int)proto->n_node, node_proto->name, node_proto->op_type);
        string node_ctx = onnxConcatCtx(ctx, format("when parsing '%s' (%s)",
                                        node_proto->name, node_proto->op_type));

        node_inputs.clear();
        node_outputs.clear();

        for (size_t j = 0; j < node_proto->n_input; j++) {
            string_view input_name = node_proto->input[j];
            if (!net->haveArg(input_name)) {
                onnxParseError(node_ctx, format("cannot find input '%s'", input_name.data()));
            }
            node_inputs.push_back(net->getArg(input_name));
        }

        for (size_t j = 0; j < node_proto->n_output; j++) {
            Arg out = net->getArg(node_proto->output[j]);
            node_outputs.push_back(out);
        }

        auto domain_it = alldispatch.find(node_proto->domain);
        bool ok = false;
        if (domain_it != alldispatch.end()) {
            auto& dispatch = domain_it->second;
            auto it = dispatch.find(node_proto->op_type);
            if (it != dispatch.end()) {
                (this->*(it->second))(node_ctx, node_proto, graph, node_inputs, node_outputs);
                ok = true;
            }
        }
        if (!ok)
            unsupportedOps.insert(node_proto->op_type);
    }

    return graph;
}

OnnxImporter2::OnnxImporter2(Net2& net, const char *fileName)
{
    init(net);
    parse(fileName);
}

OnnxImporter2::OnnxImporter2(Net2& net, const char* buffer, size_t bufsize)
{
    init(net);
    parse(buffer, bufsize);
}

bool OnnxImporter2::parse(const char* filename_)
{
    size_t fsize, freaded;
    AutoBuffer<char> buf;
    char* bufptr;

    filename = filename_;
    FILE* f = fopen(filename_, "rb");

    if (!f) {
        CV_LOG_DEBUG(NULL, "DNN/Onnx: cannot open file " << filename_);
        return false;
    }
    fseek(f, 0, SEEK_END);
    fsize = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    buf.allocate(fsize+256);
    bufptr = buf.data();
    freaded = fread(bufptr, 1, fsize, f);
    fclose(f);
    if (freaded != fsize) {
        CV_LOG_DEBUG(NULL, "DNN/Onnx: cannot read file " << filename_);
        return false;
    }
    return parse(bufptr, freaded);
}

bool OnnxImporter2::parse(const char* buffer, size_t datasize)
{
    bool ok = true;
    if (datasize == 0) {
        CV_LOG_DEBUG(NULL, "DNN/Onnx: the file/buffer is empty");
        return false;
    }

    {
        const uint8_t* data = (const uint8_t*)buffer;
        OpenCVOnnx__ModelProto* model = opencv_onnx__model_proto__unpack(0, datasize, data);
        if(model == 0) {
            CV_LOG_DEBUG(NULL, "DNN/Onnx: could not parse the model" << (filename.empty() ? " " + filename : ""));
            return false;
        }

        OnnxInfo info;
        info.IRVersion = model->ir_version;
        info.producer = model->producer_name;
        info.domain = model->domain;
        info.docString = model->doc_string;

        onnxParseArray(net, string(), model->opset_import, model->n_opset_import, info.opsets);
        net->setOnnxInfo(info);

        if (model->producer_name)
            frameworkName = model->producer_name;
        try {
            parseGraph(model->graph, false);
        } catch (const std::exception& exn) {
            string filectx = filename.empty() ? "" : " (" + filename + ") ";
            CV_LOG_WARNING(NULL, "DNN/Onnx: parse error" + filectx + exn.what());
            ok = false;
        }
        opencv_onnx__model_proto__free_unpacked(model, 0);
    }

    if (!unsupportedOps.empty()) {
        std::stringstream msg;
        bool plural = unsupportedOps.size() > 1, first = true;
        msg << string("DNN/Onnx: the operation") + (plural ? "s " : " ");
        for (string_view opname: unsupportedOps) {
            msg << (first ? "'" : ", '") + string(opname) + "'";
            first = false;
        }
        msg << (plural ? " are not supported" : " is not supported");
        CV_LOG_ERROR(NULL, msg.str());
        ok = false;
    }
    if (!ok) net->release();
    return ok;
}

void OnnxImporter2::init(Net2& net)
{
    DispatchMap dispatch, msdispatch;

    dispatch["Equal"] = dispatch["Greater"] = dispatch["Less"] = dispatch["Pow"] = dispatch["Add"] =
            dispatch["Sub"] = dispatch["Mul"] = dispatch["Div"] = &OnnxImporter2::parseElemwiseBinary;
    //dispatch["ArgMax"] = dispatch["ArgMin"] = &OnnxImporter2::parseArg;
    dispatch["AveragePool"] = &OnnxImporter2::parsePooling;
    dispatch["BatchNormalization"] = &OnnxImporter2::parseBatchNormalization;
    dispatch["Cast"] = &OnnxImporter2::parseCast;
    dispatch["Clip"] = &OnnxImporter2::parseClip;
    dispatch["Concat"] = &OnnxImporter2::parseConcat;
    dispatch["Constant"] = &OnnxImporter2::parseConstant;
    dispatch["ConstantFill"] = dispatch["ConstantOfShape"] = &OnnxImporter2::parseConstantOfShape;
    dispatch["Conv"] = &OnnxImporter2::parseConv;
    dispatch["ConvTranspose"] = &OnnxImporter2::parseConvTranspose;
    dispatch["CumSum"] = &OnnxImporter2::parseCumSum;
    //dispatch["DetectionOutput"] = &OnnxImporter2::parseDetectionOutput;
    dispatch["Dropout"] = &OnnxImporter2::parseDropout;
    dispatch["Expand"] = &OnnxImporter2::parseExpand;
    dispatch["Flatten"] = &OnnxImporter2::parseFlatten;
    dispatch["Gather"] = &OnnxImporter2::parseGather;
    dispatch["Gemm"] = &OnnxImporter2::parseGemm;
    dispatch["GlobalAveragePool"] = dispatch["GlobalMaxPool"] = &OnnxImporter2::parseGlobalPool;
    //dispatch["GRU"] = &OnnxImporter2::parseGRU;
    //dispatch["ImageScaler"] = &OnnxImporter2::parseImageScaler;
    //dispatch["InstanceNormalization"] = &OnnxImporter2::parseInstanceNormalization;
    dispatch["LeakyRelu"] = &OnnxImporter2::parseLeakyRelu;
    dispatch["LRN"] = &OnnxImporter2::parseLRN;
    //dispatch["LSTM"] = &OnnxImporter2::parseLSTM;
    dispatch["MatMul"] = &OnnxImporter2::parseMatMul;
    dispatch["Max"] = dispatch["Min"] = dispatch["Sum"] = &OnnxImporter2::parseElemwiseNary;
    dispatch["MaxPool"] = &OnnxImporter2::parsePooling;
    //dispatch["MaxUnpool"] = &OnnxImporter2::parseMaxUnpool;
    //dispatch["Pad"] = &OnnxImporter2::parsePad;
    dispatch["PRelu"] = &OnnxImporter2::parsePRelu;
    dispatch["ReduceMax"] = dispatch["ReduceMin"] = dispatch["ReduceMean"] = dispatch["ReduceSum"] = dispatch["ReduceMax"] =
    dispatch["ReduceMin"] = dispatch["ReduceSumSquare"] = dispatch["ReduceProd"] = dispatch["ReduceL1"] =
    dispatch["ReduceL2"] = dispatch["ReduceLogSum"] = dispatch["ReduceLogSumExp"] = &OnnxImporter2::parseReduce;
    dispatch["Relu"] = &OnnxImporter2::parseRelu;
    dispatch["Reshape"] = &OnnxImporter2::parseReshape;
    dispatch["Resize"] = &OnnxImporter2::parseResize;
    dispatch["Shape"] = &OnnxImporter2::parseShape;
    dispatch["Slice"] = &OnnxImporter2::parseSlice;
    dispatch["Softmax"] = dispatch["SoftMax"] = dispatch["LogSoftmax"] = &OnnxImporter2::parseSoftMax;
    dispatch["SpaceToDepth"] = dispatch["DepthToSpace"] = &OnnxImporter2::parseDepthToSpace;
    dispatch["Split"] = &OnnxImporter2::parseSplit;
    dispatch["Squeeze"] = &OnnxImporter2::parseSqueeze;
    dispatch["Transpose"] = &OnnxImporter2::parseTranspose;
    //dispatch["Upsample"] = &OnnxImporter2::parseUpsample;
    dispatch["Unsqueeze"] = &OnnxImporter2::parseUnsqueeze;

    vector<string> simpleLayers{
        "Abs", "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh",
        "Ceil", "Celu", "Cos", "Cosh", "Elu", "Erf",
        "Exp", "Floor", "HardSigmoid", "HardSwish", "Identity",
        "Log", "Neg", "Round", "Reciprocal", "Selu",
        "Sign", "Sigmoid", "Sin", "Sinh", "Softplus",
        "Softsign", "Shrink", "Sqrt", "Tan", "Tanh", "ThresholdedRelu"};
    for (const auto& name : simpleLayers)
        dispatch[name] = &OnnxImporter2::parseElemwiseUnary;

    /*
    // ai.onnx: opset 10+
    dispatch["DequantizeLinear"] = &OnnxImporter2::parseDequantizeLinear;
    dispatch["QLinearConv"] = &OnnxImporter2::parseQLinearConv;
    dispatch["QLinearMatMul"] = &OnnxImporter2::parseQLinearMatMul;
    dispatch["QuantizeLinear"] = &OnnxImporter2::parseQuantizeLinear;*/
    alldispatch[defaultOnnxDomain] = dispatch;
    alldispatch[""] = dispatch;

    /*msdispatch["QLinearAdd"] = &OnnxImporter2::parseQLinearElemwiseBinary;
    msdispatch["QLinearAveragePool"] = &OnnxImporter2::parsePooling;
    msdispatch["QLinearGlobalAveragePool"] = &OnnxImporter2::parseQLinearGlobalAveragePool;
    msdispatch["QLinearConcat"] = &OnnxImporter2::parseQLinearConcat;
    msdispatch["QLinearLeakyRelu"] = &OnnxImporter2::parseQLinearLeakyRelu;
    msdispatch["QLinearSigmoid"] = &OnnxImporter2::parseQLinearSigmoid;
    alldispatch["com.microsoft"] = msdispatch;
     */
}

void OnnxImporter2::parseBatchNormalization(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                            const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t noutputs = outputs.size();
    OnnxAssert(ctx, noutputs == 1);
    if (inputs.size() != 5)
        onnxParseError(ctx, "Expected input, scale, bias, mean and var");

    double epsilon = onnxAttrFloat(ctx, node_proto, "epsilon", 1e-5f);

    batchNorm(graph, node_proto->name, node_proto->output[0],
              inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
              epsilon);
}

void OnnxImporter2::parseCast(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                              const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    int datatype = onnxAttrInt(ctx, node_proto, "to", -1);
    OnnxAssert(ctx, datatype >= 0);
    int typ = onnxDatatypeToDepth(ctx, datatype);

    cast(graph, node_proto->name, node_proto->output[0], inputs[0], typ);
}

void OnnxImporter2::parseClip(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                              const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 1 || ninputs == 3);
    OnnxAssert(ctx, outputs.size() == 1);
    layerParams.type = "ReLU6";
}

void OnnxImporter2::parseConcat(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() >= 1);
    OnnxAssert(ctx, outputs.size() == 1);

    bool haveAxis = false;
    int axis = onnxAttrInt(ctx, node_proto, "axis", -1, &haveAxis);
    OnnxAssert(ctx, haveAxis);

    concat(graph, node_proto->name, node_proto->output[0], inputs, axis);
}


void OnnxImporter2::parseConstant(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                  const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    Tensor t;
    TensorSize scalar_shape;
    OnnxAssert(ctx, inputs.size() == 0);
    OnnxAssert(ctx, outputs.size() == 1);
    if (layerParams.has("value")) {
        DictValue v = layerParams.get("value");
        auto m = v.getMat();
        t = Tensor(m.first, m.second, true);
    } else if (layerParams.has("value_int")) {
        int v = saturate_cast<int>(layerParams.get<int64>("value_int"));
        t = Tensor(scalar_shape, CV_32S, &v, true);
    } else if (layerParams.has("value_float")) {
        float v = (float)layerParams.get<double>("value_float");
        t = Tensor(scalar_shape, CV_32F, &v, true);
    } else if (layerParams.has("value_ints")) {
        vector<int> v = layerParams.get<vector<int> >("value_ints");
        TensorSize shape;
        shape.ndims = 1;
        shape.size[0] = (int64_t)v.size();
        t = v.empty() ? Tensor() : Tensor(shape, CV_32S, &v[0], true);
    } else if (layerParams.has("value_floats")) {
        vector<float> v = layerParams.get<vector<float> >("value_floats");
        TensorSize shape;
        shape.ndims = 1;
        shape.size[0] = (int64_t)v.size();
        t = v.empty() ? Tensor() : Tensor(shape, CV_32F, &v[0], true);
    } else {
        onnxParseError(ctx, "invalid/unsupported constant type");
    }
    net->addConstTensor("", t, outputs[0]);
}

void OnnxImporter2::parseConstantOfShape(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                         const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    
}

void OnnxImporter2::parseConv(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                              const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || ninputs == 3);
    OnnxAssert(ctx, outputs.size() == 1);
    layerParams.type = "Convolution";
    bool const_conv = net->isConst(inputs[1]) && (ninputs == 2 || net->isConst(inputs[2]));
    if (const_conv) {
        for (int i = 1; i < ninputs; i++) {
            Mat t = net->tensors.at(inputs[i]).getMat();
            layerParams.blobs.push_back(t);
        }
        inputs.resize(1);
    }
}

void OnnxImporter2::parseConvTranspose(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                       const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || ninputs == 3);
    OnnxAssert(ctx, outputs.size() == 1);
    layerParams.type = "Deconvolution";
}

void OnnxImporter2::parseDropout(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                 const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size(), noutputs = outputs.size();
    OnnxAssert(ctx, 1 <= ninputs && ninputs <= 3);
    OnnxAssert(ctx, 1 <= noutputs && noutputs <= 2);
    if (noutputs == 2)
        outputs.pop_back();
}

// "Equal" "Greater" "Less" "Pow" "Add" "Sub" "Mul" "Div" ...
void OnnxImporter2::parseElemwiseBinary(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                        const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 2 && outputs.size() == 1);
    layerParams.set("op", toUpperCase(layerParams.type));
    layerParams.type = "NaryEltwise";
}

// "Sum" "Min" "Max"
void OnnxImporter2::parseElemwiseNary(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                      const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() >= 2 && outputs.size() == 1);
    layerParams.set("op", toUpperCase(layerParams.type));
    layerParams.type = "NaryEltwise";
}

void OnnxImporter2::parseElemwiseUnary(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                       const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    layerParams.set("op", layerParams.type);
    layerParams.type = "Elemwise";
}

void OnnxImporter2::parseExpand(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 2 && outputs.size() == 1);
}

void OnnxImporter2::parseFlatten(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                 const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
}

void OnnxImporter2::parseGather(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 2);
    OnnxAssert(ctx, outputs.size() == 1);
}

// A * B + C = Y, we require that the dimension of A is [m, k], and the dimension of B is [n, k].
// And the dim of output Y is [m, n]
void OnnxImporter2::parseGemm(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                              const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || ninputs == 3);
    layerParams.type = "InnerProduct";
}

void OnnxImporter2::parseGlobalPool(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                    const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    layerParams.type = "Pooling";
    string pool;
    if (strcmp(node_proto->op_type, "GlobalMaxPool") == 0)
        pool = "MAX";
    else if (strcmp(node_proto->op_type, "GlobalAveragePool") == 0)
        pool = "AVE";
    else
        onnxParseError(ctx, format("Unsupported pooling operation '%s'", node_proto->op_type));

    layerParams.set("global_pooling", true);
    layerParams.set("pool", pool);
}


void OnnxImporter2::parseLeakyRelu(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                   const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 2);
    layerParams.type = "ReLU";
    layerParams.set("negative_slope", layerParams.get<float>("alpha", 0.01));
}

void OnnxImporter2::parseLRN(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                             const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    replaceLayerParam(layerParams, "size", "local_size");
}

void OnnxImporter2::parseMatMul(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 2 && outputs.size() == 1);
    layerParams.type = "Gemm";
}


void OnnxImporter2::parsePooling(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                 const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    bool quantized = strcmp(node_proto->op_type, "QLinearAveragePool") == 0;
    string pool_type = strcmp(node_proto->op_type, "AveragePool") == 0 ? "AVE" :
                       strcmp(node_proto->op_type, "MaxPool") == 0 ? "MAX" :
                       strcmp(node_proto->op_type, "QLinearAveragePool") == 0 ? "AVE" : "";
    layerParams.type = "Pooling";
    layerParams.set("pool", pool_type);
    replaceLayerParam(layerParams, "kernel_shape", "kernel_size");
    // auto_pad attribute is deprecated and uses ceil
    if (layerParams.has("pad_mode"))
        layerParams.set("ceil_mode", true);
    else if (!layerParams.has("ceil_mode"))
        layerParams.set("ceil_mode", false);
    layerParams.set("ave_pool_padded_area", frameworkName == "pytorch");
    size_t ninputs = inputs.size(), noutputs = outputs.size();
    if (quantized) {
        OnnxAssert(ctx, ninputs == 4 || ninputs == 5);
    } else {
        OnnxAssert(ctx, ninputs == 1);
    }
    OnnxAssert(ctx, noutputs == 1);
}

void OnnxImporter2::parsePRelu(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                               const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 2 && outputs.size() == 1);
    layerParams.type = "PReLU";
}

void OnnxImporter2::parseReduce(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    string op_type = node_proto->op_type;
    string reduceType;

    if (op_type == "ReduceMax")
        reduceType = "MAX";
    else if (op_type == "ReduceMin")
        reduceType = "MIN";
    else if (op_type == "ReduceSum")
        reduceType = "SUM";
    else if (op_type == "ReduceSumSquare")
        reduceType = "SUM_SQUARE";
    else if (op_type == "ReduceProd")
        reduceType = "PROD";
    else if (op_type == "ReduceL1")
        reduceType = "L1";
    else if (op_type == "ReduceL2")
        reduceType = "L2";
    else if (op_type == "ReduceLogSum")
        reduceType = "LOG_SUM";
    else if (op_type == "ReduceLogSumExp")
        reduceType = "LOG_SUM_EXP";
    else if (op_type == "ReduceMean")
        reduceType = "AVE";
    else
        onnxParseError(ctx, format("unsupported reduce operation '%s'", node_proto->op_type));

    layerParams.type = "Reduce";
    layerParams.set("reduce", reduceType);
}

void OnnxImporter2::parseRelu(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                              const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    layerParams.type = "ReLU";
    layerParams.set("op", "ReLU");
}

void OnnxImporter2::parseReshape(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                 const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || (ninputs == 1 && layerParams.has("shape")));
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseResize(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, 1 <= ninputs && ninputs <= 4);
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseShape(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                               const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1);
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseSlice(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                               const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, 3 <= ninputs && ninputs <= 5);
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseSoftMax(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                 const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1);
    OnnxAssert(ctx, outputs.size() == 1);
    layerParams.set("log_softmax", layerParams.type == "LogSoftmax");
    layerParams.type = "Softmax";
}

void OnnxImporter2::parseSplit(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                               const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1);
    OnnxAssert(ctx, outputs.size() >= 1);
    if (layerParams.has("num_split")) {
        int num_split = layerParams.get<int>("num_split");
        OnnxAssert(ctx, num_split == outputs.size());
    }
}

void OnnxImporter2::parseSqueeze(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                 const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || (ninputs == 1 && layerParams.has("axes")));
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseTranspose(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                   const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    OnnxAssert(ctx, inputs.size() == 1 && outputs.size() == 1);
    OnnxAssert(ctx, layerParams.has("perm"));
}

void OnnxImporter2::parseUnsqueeze(string_view ctx, const OpenCVOnnx__NodeProto* node_proto, Graph& graph,
                                   const vector<Arg>& inputs, const vector<Arg>& outputs)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || (ninputs == 1 && layerParams.has("axes")));
    OnnxAssert(ctx, outputs.size() == 1);
}

/*void OnnxImporter2::parseDequantizeLinear(string_view ctx,
                                          const OpenCVOnnx__NodeProto* node_proto, Graph& graph)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || ninputs == 3);
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseQLinearConcat(string_view ctx,
                                       const OpenCVOnnx__NodeProto* node_proto, Graph& graph)
{
    OnnxAssert(ctx, inputs.size() >= 3);
    OnnxAssert(ctx, outputs.size() == 1);
    OnnxAssert(ctx, layerParams.has("axis"));
}

void OnnxImporter2::parseQLinearConv(string_view ctx,
                                     const OpenCVOnnx__NodeProto* node_proto, Graph& graph)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 8 || ninputs == 9);
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseQLinearElemwiseBinary(string_view ctx,
                                               const OpenCVOnnx__NodeProto* node_proto, Graph& graph)
{
    size_t ninputs = inputs.size();
    string op = layerParams.type == "QLinearAdd" ? "ADD" :
                layerParams.type == "QlinearMul" ? "MUL" : "";
    if (op.empty())
        onnxParseError(ctx, format("unrecognized quantized binary operation '%s'", node_proto->op_type));
    OnnxAssert(ctx, ninputs == 7 || ninputs == 8);
    OnnxAssert(ctx, outputs.size() == 1);
    layerParams.type = "QElemwise";
    layerParams.set("op", op);
}

void OnnxImporter2::parseQLinearGlobalAveragePool(string_view ctx,
                                       const OpenCVOnnx__NodeProto* node_proto, Graph& graph)
{
    OnnxAssert(ctx, inputs.size() == 5);
    OnnxAssert(ctx, outputs.size() == 1);
    OnnxAssert(ctx, layerParams.has("channels_last"));
}

void OnnxImporter2::parseQLinearLeakyRelu(string_view ctx,
                                          const OpenCVOnnx__NodeProto* node_proto, Graph& graph)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 4 || ninputs == 5);
    OnnxAssert(ctx, outputs.size() == 1);
    OnnxAssert(ctx, layerParams.has("alpha"));
}

void OnnxImporter2::parseQLinearMatMul(string_view ctx,
                                       const OpenCVOnnx__NodeProto* node_proto, Graph& graph)
{
    OnnxAssert(ctx, inputs.size() == 8);
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseQLinearSigmoid(string_view ctx,
                                        const OpenCVOnnx__NodeProto* node_proto, Graph& graph)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 4 || ninputs == 5);
    OnnxAssert(ctx, outputs.size() == 1);
}

void OnnxImporter2::parseQuantizeLinear(string_view ctx,
                                        const OpenCVOnnx__NodeProto* node_proto, Graph& graph)
{
    size_t ninputs = inputs.size();
    OnnxAssert(ctx, ninputs == 2 || ninputs == 3);
    OnnxAssert(ctx, outputs.size() == 1);
}*/

Net2 readNetFromONNX2(string_view onnxFile)
{
    Net2 net;
    OnnxImporter2 importer(net, onnxFile.c_str());
    Ptr<Net2::Impl> net = net.impl();
    if (net.empty())
        return net;
    net.impl()->fuse();
    net.impl()->assignBuffers();
    return net;
}

}} // namespace
