// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"
#include "gtest/gtest.h"
#include "test_utils.h"
#include "ocos.h"


struct Input {
  const char* name = nullptr;
  std::vector<int64_t> dims;
  std::vector<float> values;
};

auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_VERBOSE, "Default");

void RunSession(Ort::Session& session_object,
                const std::vector<Input>& inputs,
                const char* output_name,
                const std::vector<int64_t>& dims_y,
                const std::vector<float>& values_y) {

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(memory_info, 
      const_cast<float*>(inputs[i].values.data()), inputs[i].values.size(), inputs[i].dims.data(), inputs[i].dims.size()));
  }

  std::vector<Ort::Value> ort_outputs;
  ort_outputs = session_object.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(), ort_inputs.size(), &output_name, 1);
  ASSERT_EQ(ort_outputs.size(), 1u);
  auto output_tensor = &ort_outputs[0];

  auto type_info = output_tensor->GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), dims_y);
  size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(values_y.size(), total_len);

  float* f = output_tensor->GetTensorMutableData<float>();
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(values_y[i], f[i]);
  }
}

void TestInference(Ort::Env& env, const char* model_uri,
                   const std::vector<Input>& inputs,
                   const char* output_name,
                   const std::vector<int64_t>& expected_dims_y,
                   const std::vector<float>& expected_values_y,
                   const char* custom_op_library_filename) {
  Ort::SessionOptions session_options;
  if (custom_op_library_filename) {
    Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary((OrtSessionOptions*)session_options, custom_op_library_filename, nullptr));
  }

  // if session creation passes, model loads fine
  Ort::Session session(env, model_uri, session_options);

  // Now run
  RunSession(session,
              inputs,
              output_name,
              expected_dims_y,
              expected_values_y);
}


TEST(utils, test_ort_case) {
  
  std::cout << "Running custom op inference" << std::endl;

  std::vector<Input> inputs(1);
  Input& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

#if defined(_WIN32)
  const char lib_name[] = "custom_op_library.dll";
#elif defined(__APPLE__)
  const char lib_name[] = "libcustom_op_library.dylib";
#else
  const char lib_name[] = "./libcustom_op_library.so";
#endif
  TestInference(*ort_env, "test/data/custom_op_test.onnx", inputs, "Y", expected_dims_y, expected_values_y, lib_name);
}
