#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <vector>

void run_attention_kernel(float* Q, float* K, float* V, float* Output, size_t size) {
    // Load binary
    std::ifstream bin_file("attention_kernel.xclbin", std::ios::binary);
    std::vector<unsigned char> binary((std::istreambuf_iterator<char>(bin_file)),
                                       std::istreambuf_iterator<char>());
    
    // Create context, queue, and program
    cl::Context context(CL_DEVICE_TYPE_ACCELERATOR);
    cl::CommandQueue queue(context);
    cl::Program program(context, {binary});

    // Load kernel
    cl::Kernel kernel(program, "attention_kernel");

    // Allocate buffers
    cl::Buffer q_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, Q);
    cl::Buffer k_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, K);
    cl::Buffer v_buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, V);
    cl::Buffer out_buf(context, CL_MEM_WRITE_ONLY, size);

    // Set arguments
    kernel.setArg(0, q_buf);
    kernel.setArg(1, k_buf);
    kernel.setArg(2, v_buf);
    kernel.setArg(3, out_buf);

    // Launch kernel
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(out_buf, CL_TRUE, 0, size, Output);
}