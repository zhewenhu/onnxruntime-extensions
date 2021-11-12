import onnxruntime as ort
import numpy as np
import time, os
from onnxruntime_extensions import get_library_path
from dataset_loading import mnist

trainx, trainy, test_imgs, test_labels, valx, valy = mnist.load_mnist_data(
    "/tmp", download=True, one_hot=False
)

ok = 0
nok = 0
total = test_imgs.shape[0]

so = ort.SessionOptions()
so.register_custom_ops_library(get_library_path())

sess = ort.InferenceSession("tfc_w1_a1_streamlined.onnx", so)

inname = [input.name for input in sess.get_inputs()]
outname = [output.name for output in sess.get_outputs()]

start_time = time.time()

for i in range(total):
    exp = test_labels[i]

    inputs = test_imgs[i][np.newaxis, :, :, :].astype(np.float32)
    inputs = np.transpose(inputs, (0, 3, 1, 2))

    results = sess.run(outname, {inname[0]: inputs})[0]

    ok += exp == results
    nok += exp != results

end_time = time.time()
runtime = end_time - start_time
acc = 100.0 * ok / (total)
print("Accuracy: %f" % acc)
print("Runtime (s): {}".format(runtime))