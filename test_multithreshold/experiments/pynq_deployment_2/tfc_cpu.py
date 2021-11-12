import onnxruntime as ort
import numpy as np
from onnxruntime_extensions import get_library_path

so = ort.SessionOptions()
so.register_custom_ops_library(get_library_path())

sess = ort.InferenceSession("tfc_w1_a1_streamlined.onnx", so)

inname = [input.name for input in sess.get_inputs()]
outname = [output.name for output in sess.get_outputs()]

inputs = np.load("input.npy")

results = sess.run(outname, {inname[0]: inputs})[0]
print(results)