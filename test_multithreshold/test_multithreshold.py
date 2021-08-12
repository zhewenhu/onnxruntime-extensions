import onnxruntime as ort
import numpy as np
from onnxruntime_extensions import get_library_path

so = ort.SessionOptions()
so.register_custom_ops_library(get_library_path())

sess = ort.InferenceSession("test.onnx", so)

inname = [input.name for input in sess.get_inputs()]
outname = [output.name for output in sess.get_outputs()]

print("Inputs name:", inname)
print("Output name:", outname)

inputs = np.ndarray(
        shape=(6, 3, 2, 2),
        buffer=np.array(
            [
                4.8,
                3.2,
                1.2,
                4.9,
                7.8,
                2.4,
                3.1,
                4.7,
                6.2,
                5.1,
                4.9,
                2.2,
                6.2,
                0.0,
                0.8,
                4.7,
                0.2,
                5.6,
                8.9,
                9.2,
                9.1,
                4.0,
                3.3,
                4.9,
                2.3,
                1.7,
                1.3,
                2.2,
                4.6,
                3.4,
                3.7,
                9.8,
                4.7,
                4.9,
                2.8,
                2.7,
                8.3,
                6.7,
                4.2,
                7.1,
                2.8,
                3.1,
                0.8,
                0.6,
                4.4,
                2.7,
                6.3,
                6.1,
                1.4,
                5.3,
                2.3,
                1.9,
                4.7,
                8.1,
                9.3,
                3.7,
                2.7,
                5.1,
                4.2,
                1.8,
                4.1,
                7.3,
                7.1,
                0.4,
                0.2,
                1.3,
                4.3,
                8.9,
                1.4,
                1.6,
                8.3,
                9.4,
            ]
        ),
    )
thresholds = np.ndarray(
        shape=(3, 7),
        buffer=np.array(
            [
                0.8,
                1.4,
                1.7,
                3.5,
                5.2,
                6.8,
                8.2,
                0.2,
                2.2,
                3.5,
                4.5,
                6.6,
                8.6,
                9.2,
                1.3,
                4.1,
                4.5,
                6.5,
                7.8,
                8.1,
                8.9,
            ]
        ),
    )

outputs = np.ndarray(
    shape=(6, 3, 2, 2),
    buffer=np.array(
        [
            4.0,
            3.0,
            1.0,
            4.0,
            5.0,
            2.0,
            2.0,
            4.0,
            3.0,
            3.0,
            3.0,
            1.0,
            5.0,
            0.0,
            1.0,
            4.0,
            1.0,
            4.0,
            6.0,
            7.0,
            7.0,
            1.0,
            1.0,
            3.0,
            3.0,
            3.0,
            1.0,
            3.0,
            4.0,
            2.0,
            3.0,
            7.0,
            3.0,
            3.0,
            1.0,
            1.0,
            7.0,
            5.0,
            4.0,
            6.0,
            2.0,
            2.0,
            1.0,
            1.0,
            2.0,
            1.0,
            3.0,
            3.0,
            2.0,
            5.0,
            3.0,
            3.0,
            4.0,
            5.0,
            7.0,
            3.0,
            1.0,
            3.0,
            2.0,
            1.0,
            4.0,
            6.0,
            6.0,
            0.0,
            1.0,
            1.0,
            3.0,
            6.0,
            1.0,
            1.0,
            6.0,
            7.0,
        ]
    ),
)

results = sess.run(outname, {inname[0]: inputs, inname[1]: thresholds})[0]

assert (results == outputs).all()


# python implementation for results check
def compare(x, y):
    """Comparison helper function for multithresholding.
    Gets two values and returns 1.0 if x>=y otherwise 0.0."""
    if x >= y:
        return 1.0
    else:
        return 0.0


def multithreshold_elementwise(v, thresholds, out_scale=None, out_bias=None):
    """Given a set of threshold values t={t_0, t_1 ... t_n} the successive
    thresholding maps any real number x to an integer in the interval [0, n],
    where the returned integer is the number of thresholds x is greater than
    or equal to.
    The output tensor will be scaled by out_scale and biased by out_bias."""
    # the inputs are expected to be in the shape (N,C,H,W) or (N, C)
    # the MultiThreshold node supports a data_layout attribute that can be set
    # to 'NHWC' to support (N,H,W,C) data layout mode for in-out as well
    # N : Batch size
    # C : Number of channels
    # H : Heigth of the input images
    # W : Width of the input images
    #
    # the thresholds are expected to be in the shape (C, B)
    # C : Number of channels (must be the same value as C in input tensor
    #     or 1 if all channels use the same threshold value)
    # B : Desired activation steps => i.e. for 4-bit activation,
    #     B=7 (2^(n)-1 and n=4)
    # the output tensor will be scaled by out_scale and biased by out_bias
    # assert threshold shape
    is_global_threshold = thresholds.shape[0] == 1
    assert (
        v.shape[1] == thresholds.shape[0]
    ) or is_global_threshold, """"Threshold
    shape incorrect"""
    # save the required shape sizes for the loops (N, C and B)
    num_batch = v.shape[0]
    num_channel = v.shape[1]
    num_act = thresholds.shape[1]
    # reshape inputs to enable channel-wise reading
    vr = v.reshape((v.shape[0], v.shape[1], -1))
    # save the new shape size of the images
    num_img_elem = vr.shape[2]
    # initiate output tensor
    ret = np.zeros_like(vr)
    # iterate over thresholds channel-wise
    for t in range(num_channel):
        channel_thresh = thresholds[0] if is_global_threshold else thresholds[t]
        # iterate over batches
        for b in range(num_batch):
            # iterate over image elements on which the thresholds will be applied
            for elem in range(num_img_elem):
                # iterate over the different thresholds for one channel
                for a in range(num_act):
                    # apply successive thresholding to every element
                    ret[b][t][elem] += compare(vr[b][t][elem], channel_thresh[a])
    if out_scale is None:
        out_scale = 1.0
    if out_bias is None:
        out_bias = 0.0
    return out_scale * ret.reshape(v.shape) + out_bias

# performance and random test
np.random.seed(0)
inputs = np.random.random((6, 3, 2, 2))
thresholds = (np.array([[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9]]) - 0.5) / 6

ort_results = sess.run(outname, {inname[0]: inputs, inname[1]: thresholds})[0]
py_results = multithreshold_elementwise(inputs, thresholds)

print(ort_results)
assert (ort_results == py_results).all()
