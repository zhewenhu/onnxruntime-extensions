#pragma once

#include <cassert>
#include <stdint.h>
#include "ocos.h"

// static const char* c_OpDomain = "finn.custom_op.general";

struct MultithresholdKernel {
    MultithresholdKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info) : ort_(ort) {
        out_dtype_ = ort_.KernelInfoGetAttribute<std::string>(info, "out_dtype");
        out_scale_ = ort_.KernelInfoGetAttribute<float>(info, "out_scale");
        out_bias_ = ort_.KernelInfoGetAttribute<float>(info, "out_bias");
        data_layout_ = ort_.KernelInfoGetAttribute<std::string>(info, "data_layout");
    }

    void Compute(OrtKernelContext* context);

private:
    Ort::CustomOpApi ort_;
    std::string out_dtype_;
    float out_scale_;
    float out_bias_;
    std::string data_layout_;
};

struct MultithresholdOp : Ort::CustomOpBase<MultithresholdOp, MultithresholdKernel> {

    void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) const { return new MultithresholdKernel(api, info); };
    const char* GetName() const { return "MultiThreshold"; };

    size_t GetInputTypeCount() const { return 2; };
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
        // Both the inputs need to be necessarily of float type
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    };

    size_t GetOutputTypeCount() const { return 1; };
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    };
};

void *compute_NCHW(const OrtTensorDimensions &dim_v, const OrtTensorDimensions &dim_thresh, const float* val, const float* thresh, float *result) {
    // save the shape sizes
    const int64_t num_batch = dim_v[0];
    const int64_t num_channel = dim_v[1];
    const int64_t num_height = dim_v[2];
    const int64_t num_width = dim_v[3];
    const int64_t num_img_elem = num_height * num_width;
    const int64_t num_act = dim_thresh[1];

    // assert threshold shape
    bool is_global_threshold = dim_thresh[0] == 1;
    assert(num_channel == dim_thresh.data()[0] || is_global_threshold);

    // iterate over thresholds channel-wise
    for (int t = 0; t < num_channel; ++t) {
        auto *channel_thresh = is_global_threshold ? &thresh[0] : &thresh[t*num_act];
        // iterate over batches
        for (int b = 0; b < num_batch; ++b) {
            int64_t elem_base_index = b*num_channel*num_img_elem + t*num_img_elem;
            // iterate over image elements on which the thresholds will be applied
            for (int elem = 0; elem < num_img_elem; ++elem) {
                double temp = 0.0;
                // iterate over the different thresholds for one channel
                for (int a = 0; a < num_act; ++a) {
                    if (val[elem_base_index + elem] >= channel_thresh[a])
                        temp++;
                    else
                        break;
                }
                result[elem_base_index + elem] = temp;
            }
        }
    }
    return nullptr;
}

void *compute_NHWC_NC(const OrtTensorDimensions &dim_v, const OrtTensorDimensions &dim_thresh, const float* val, const float* thresh, float *result) {
    // save the shape sizes
    const int64_t num_batch = dim_v[0];
    const int64_t num_channel = dim_v.back();
    int64_t num_pixel;
    if (dim_v.size() == 4) {
        const int64_t num_width = dim_v[2];
        const int64_t num_height = dim_v[1];
        num_pixel = num_width * num_height;

    } else {
        num_pixel = 1;
    }
    const int64_t num_all = num_batch  * num_channel * num_pixel;
    const int64_t num_act = dim_thresh[1];

    // assert threshold shape
    bool is_global_threshold = dim_thresh[0] == 1;
    assert(num_channel == dim_thresh.data()[0] || is_global_threshold);

    // iterate over thresholds channel-wise

    for (int t = 0; t < num_channel; ++t) {
        auto *channel_thresh = is_global_threshold ? &thresh[0] : &thresh[t*num_act];
        for (int64_t i = 0; i < num_all; i += num_channel) {
            double temp = 0.0;
            for (int a = 0; a < num_act; ++a) {
                if (val[i + t] >= channel_thresh[a])
                    temp++;
                else
                    break;
            }
            result[i + t] = temp;
        }
    }

    return nullptr;
}

void MultithresholdKernel::Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_v             = ort_.KernelContext_GetInput(context, 0);
    const OrtValue* input_thresholds    = ort_.KernelContext_GetInput(context, 1);

    const auto *v = ort_.GetTensorData<float>(input_v);
    const auto *thresholds = ort_.GetTensorData<float>(input_thresholds);

    // Setup output
    OrtTensorDimensions dimensions_v(ort_, input_v);
    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions_v.data(), dimensions_v.size());
    auto *out = ort_.GetTensorMutableData<float>(output);
    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    // Do computation
    assert(data_layout_ == "NCHW" || data_layout_ == "NHWC" || data_layout_ == "NC");
    OrtTensorDimensions dimensions_threshold(ort_, input_thresholds);

    if (data_layout_ == "NCHW") compute_NCHW(dimensions_v, dimensions_threshold, v,  thresholds, out);
    else compute_NHWC_NC(dimensions_v, dimensions_threshold, v,  thresholds, out);

    int64_t size = 1;
    for (int i = 0; i < dimensions_v.size(); ++i) {
        size *= dimensions_v[i];
    }

    if (out_scale_ != 1.0 || out_bias_ != 0.0) {
        for (int i = 0; i < size; ++i) {
            out[i] = out_scale_ * out[i] + out_bias_;
        }
    }
}
