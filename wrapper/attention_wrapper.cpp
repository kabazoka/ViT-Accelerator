#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../hls_source/attention_kernel.h"

namespace py = pybind11;

void run_attention_kernel(py::array_t<float> Q,
                          py::array_t<float> K,
                          py::array_t<float> V,
                          py::array_t<float> Output) {
    // Ensure inputs are contiguous and properly shaped
    auto q = Q.mutable_data();  // Writable pointer to raw data
    auto k = K.mutable_data();
    auto v = V.mutable_data();
    auto out = Output.mutable_data();

    // Ensure dimensions match expectations
    size_t num_tokens = Q.shape(0);  // Number of tokens
    size_t d_head = Q.shape(1);      // Head dimension

    // Call the FPGA kernel
    attention_kernel(q, k, v, out, num_tokens, d_head);
}

PYBIND11_MODULE(attention_wrapper, m) {
    m.def("run_attention_kernel", &run_attention_kernel, "Run Attention Kernel");
}
