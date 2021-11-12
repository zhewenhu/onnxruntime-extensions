1. Install `sudo apt install build-essential` and update CMake and GCC, tested with CMake 3.21 and GCC 10.
2. In the root folder `sudo python3 setup.py install` to build and install the package.
3. `pip3 install onnx==1.9.0 onnxruntime==1.8.1` to install ONNX and ONNX Runtime.
4. Run `model_generator.py` to generate ONNX model.
5. Run `test_multithreshold` to test.
