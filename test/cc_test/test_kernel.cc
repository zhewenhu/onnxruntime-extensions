// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <math.h>
#include "test_output.hpp"

KernelOne::KernelOne(OrtApi api) : BaseKernel(api) {
}

void KernelOne::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
  const float* X = ort_.GetTensorData<float>(input_X);
  const float* Y = ort_.GetTensorData<float>(input_Y);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);

  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  float* out = ort_.GetTensorMutableData<float>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // Do computation
  for (int64_t i = 0; i < size; i++) {
    out[i] = X[i] + Y[i];
  }
}

void* CustomOpOne::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
  return new KernelOne(api);
};

const char* CustomOpOne::GetName() const {
  return "CustomOpOne";
};

size_t CustomOpOne::GetInputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpOne::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

size_t CustomOpOne::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpOne::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

KernelTwo::KernelTwo(OrtApi api) : BaseKernel(api) {
}

void KernelTwo::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  const float* X = ort_.GetTensorData<float>(input_X);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);

  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  int32_t* out = ort_.GetTensorMutableData<int32_t>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // Do computation
  for (int64_t i = 0; i < size; i++) {
    out[i] = (int32_t)(round(X[i]));
  }
}

void* CustomOpTwo::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
  return new KernelTwo(api);
};

const char* CustomOpTwo::GetName() const {
  return "CustomOpTwo";
};

size_t CustomOpTwo::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpTwo::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

size_t CustomOpTwo::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpTwo::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
};
