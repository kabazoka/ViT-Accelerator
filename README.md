# ViT-Accelerator

This repository demonstrates how to **accelerate a Vision Transformer (ViT)** on an FPGA using **High-Level Synthesis (HLS)** and **Xilinx’s XRT** runtime. It showcases:

1. A **systolic-array–style** matrix multiplication (in HLS) for computing parts of the attention mechanism.
2. A **softmax** approximation kernel to complete the attention scoring.
3. A **host (CPU) code** that manages data transfers and kernel invocation via XRT.

Below is an overview of each major component, a brief introduction to the ViT concept, and a final study journal describing the learning process from a student’s perspective.

## Requirements

- python=3.11
  - Library dependencies are listed in host/requirements.txt
- Xilinx Vitis Toolkit=2022.1

## Usage

```bash
# Installing Python dependencies
pip install -r host/requirements.txt

# csim, csynth, and co-sim via vitis_hls
cd hls_source/
vitis_hls -f scripts/run_hls.tcl

The actual host code has not done yet.

# 1) Compile each HLS kernel
v++ -c -t hw --platform xilinx_u50_gen3x16_xdma_201920_3 \
    -k attention_kernel \
    kernel.cpp \
    -o attention_kernel.xo

v++ -c -t hw --platform xilinx_u50_gen3x16_xdma_201920_3 \
    -k mmult \
    q_matmul_k_function.cpp \
    -o mmult.xo

# 2) Link into .xclbin
v++ -l -t hw --platform xilinx_u50_gen3x16_xdma_201920_3 \
    attention_kernel.xo \
    mmult.xo \
    -o vit_accel.xclbin

# Then in host code, load `vit_accel.xclbin` and call the kernels.
# Building host code
cd host/
mkdir build && cd build

# (OPTIONS) IF NOT USING FPGA:
cmake .. -DUSE_FPGA=OFF
# IF USING FPGA:
cmake .. -DUSE_FPGA=ON

cmake --build . --config Release
./bin/vit -t 4 -m ../ggml-model-f16.gguf -i ../cat-resized.jpg

```

## Declaration

The C++ host code was originally authored by https://github.com/staghado and was subsequently copied from his repository: https://github.com/staghado/vit.cpp. All credit for this code is due to the original author.

---

## 1. Introduction to Vision Transformers (ViT)

A **Vision Transformer** (ViT) applies the **Transformer** architecture—originally popularized for natural language processing—to image tasks. Key highlights:

- **Patch Embedding**: An input image is split into smaller patches; each patch is linearly projected to form an embedding.  
- **Attention Mechanism**: Instead of scanning the entire image with convolution, the ViT attends to patches via **self-attention**. In simplified terms:
  \[
    \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V}.
  \]
- **Transformer Encoder**: A stack of multi-head self-attention layers plus feed-forward networks, normalizations, etc.

While powerful, **attention** can be computationally heavy (\(\mathbf{Q}\times \mathbf{K}^T\) is a large matrix multiplication), motivating hardware acceleration.

![Vision Transformer](image/ViT.gif)

---

## 2. HLS Attention Kernel

### 2.1 Kernel Overview

Two main HLS kernels are demonstrated:

1. **`attention_kernel`** (in `kernel.cpp`):  
   - Reads \(\mathbf{Q}\) and \(\mathbf{K}\) from global memory (AXI).  
   - Multiplies them (via a sub-routine `mmult(...)` or an internal HLS-optimized approach).  
   - Applies scaling by \(\frac{1}{\text{SCAILING_FACTOR}}\).  
   - Approximates the **softmax** by summation and polynomial exponent.  
   - Writes the resulting attention scores to global memory.

2. **`mmult`** or **q_matmul_k_function** (systolic array approach):  
   - A **tiled** approach with zero-padding to handle 197×197 operations, performed in 16×16 blocks.  
   - Uses shift registers internally to mimic a **systolic array**: data are loaded in a pipeline, and partial sums accumulate in local buffers.

### 2.2 Key HLS Pragmas

- `#pragma HLS INTERFACE m_axi port=...` for AXI data transfer from host.  
- `#pragma HLS PIPELINE II=1` to pipeline loops for better throughput.  
- `#pragma HLS ARRAY_PARTITION` to create multiple memory ports for sub-blocks, enabling parallel read/writes.  
- Tiling logic (16×16 blocks) ensures we can handle large dimension (197×197) without needing huge on-chip buffers.

### 2.3 Flow in `attention_kernel`

1. **Load** matrices \(\mathbf{Q}\) and \(\mathbf{K}\) into local 2D arrays.  
2. **Call** `mmult(...)` to compute \(\mathbf{Q}\times \mathbf{K}\).  
3. **Scale** the results by a factor (e.g., dividing by 8).  
4. **Softmax** each row: 
   - Use an approximate exponential function `exp_approx(x) = 1 + x + x^2/2 + x^3/6`.  
   - Normalize by summing partial exponent results.  
5. **Write** the final attention scores back to global memory.

---

## 3. XRT Host Code

### 3.1 Host Overview

The **host** program (CPU side) is responsible for:

- **Loading the `.xclbin`** bitstream onto the FPGA device (Alveo U50 or similar).  
- Creating **buffers** (`xrt::bo`) for input \(\mathbf{Q}, \mathbf{K}\) and output attention scores.  
- **Setting kernel arguments** and **launching** the HLS kernel.  
- Optionally, providing CPU fallback code if no FPGA is present (depends on compile flags).

---

## 6. Hao Chen's Study Journal

> **Project**: Accelerating Vision Transformer Attention with FPGA

### 6.1 What I Did

I combined:

- **HLS kernels** (`attention_kernel`, `mmult`) in C/C++.  
- A **host program** using XRT to load the `.xclbin` and manage data buffers.  
- **ViT** basics: focusing on the attention step, which multiplies query (\(\mathbf{Q}\)) and key (\(\mathbf{K}\)) matrices, then applies scaling and softmax.

### 6.2 What I Learned

1. **ViT Attention**:  
   I discovered that self-attention is mostly matrix multiplications + a softmax normalization. The shape can be 197×64 or 197×197 for typical image patches.  
   - Realizing that the HPC challenge is \(\mathbf{Q}\times \mathbf{K}\) for each head.

2. **HLS Pragmas**:  
   - `#pragma HLS INTERFACE m_axi` attaches top-level arguments to AXI memory, so the host can transfer input/output.  
   - `#pragma HLS PIPELINE II=1` helps achieve one iteration per cycle in loops.  
   - Tiling in 16×16 blocks is crucial for **resource usage** and **performance** on the FPGA.

3. **Systolic Array Logic**:  
   - Using shift registers inside loops, we can replicate a specialized hardware matrix multiplication pipeline.  
   - Zero-padding ensures dimension mismatches (like 197 not being a multiple of 16) don’t break the logic.

4. **Softmax Approximation**:  
   - A polynomial expansion for \(e^x\) (`1 + x + x^2/2 + x^3/6`) saves hardware resources and latency vs. a full floating exponent.  
   - Summation is needed for normalization.

5. **XRT Host Flow**:  
   - We will try compile each kernel into `.xo`.  
   - Link them into `.xclbin`.  
   - In the host code, we do `device.load_xclbin(...)`, `xrt::kernel(...)`, create buffers, set arguments, run.  
   - The synergy of CPU code + HLS kernel is reminiscent of CPU/GPU programming, except we’re explicitly loading the bitstream onto the FPGA.

6. **Overall Project Understanding**:
   - A real-world FPGA acceleration project involves **both** the hardware design (HLS kernels) **and** the software integration (host code with XRT).  
   - The Vision Transformer uses attention heavily, so accelerating \(\mathbf{Q}\times \mathbf{K}^T\) and softmax can significantly speed up inference.

### 6.3 Challenges & Future Steps

- Handling edge cases with zero-padding or partial tiles was tricky.  
- In the future, it might be interesting to integrate the entire ViT forward pass on the FPGA, or replicate multiple attention kernels for multi-head parallelism.

## 7. Sheng-Wei Huang's Study Journal

### 7.1 What I Did

- Design an Systolic Array HLS Kernel (mmult) for matrix multiplication between the query and key matrices in Self-Attention.
- In the Vision Transformer, where the input matrix has dimensions of (224, 224, 3), the query and key matrices in Self-Attention have dimensions of (197, 64, 12) and (64, 197, 12), respectively.The last dimension of the query and key matrices represents the number of heads in Multi-Head Self-Attention, meaning 12 heads are processed independently.For this Final Project, the Systolic Array HLS Kernel is specifically designed for the matrix multiplication of one head, handling matrix dimensions of 197 × 64 and 64 × 197.
- By connecting Local Buffer and the AXI4 Master Protocol, the design allows data to be fetched from the FPGA's Global Memory and provided for computation by the Systolic Array HLS Kernel.

### 7.2 Challenges & Future Steps

- Since the current Systolic Array can only handle fixed-size matrix computations, future improvements will focus on modifying it to support matrix computations of arbitrary sizes.
- In the current design for Memory Access, data is first fetched from Global Memory via the AXI4 Master Protocol and stored in the Local Buffer before being loaded into the Sub Buffer for computation by the Systolic Array HLS Kernel. In the future, the Sub Buffer will be directly connected to the AXI4 Master Protocol to retrieve data, reducing the access time for storing data from Global Memory to the Local Buffer in the overall architecture.

## 8. Hua-Shao Chu's Study Journal

### 8.1 What I Did

- I focus on optimizing the **softmax function**. Since the softmax function requires an exponential function, either use an approximation or use the built-in function `hls::exp()`.
> [!NOTE]
> While using `hls::exp()`, you'll need to `#include <hls_math.h>`.

> [!WARNING]
> For using `hls::exp()`, the **INTERNAL-INFO: never saw llvm instructions 'fexp'(507)** message will be displayed in the c synthesis step but will not affect the results.
  
- The first step of the softmax function is to calculate the exponents of the input elements and accumulate them together, but during the accumulation operation, an II violation occurs between "load" and "store", so I increase the distance to make II=1.
  ```cpp
  for (int j = 0; j < REAL_K_COL; j++) {
      sum[j % 8] += exp_approx(scores[i][j]);
  }
  for (int j = 1; j < 8; j++) {
      sum[0] += sum[j];
  }
  ```

### 8.2 Challenges & Future Steps

Although implementing the softmax function is not very difficult, optimizing it requires many factors to be considered.
- **Resource Usage:** Since both inputs and outputs are huge matrices, using just one instance will take many times to complete, so there is a trade-off between latency and area.
- **Pipeline strategy:** In order to improve hardware utilization and do not want to leave the hardware idle, the pipeline design must be optimized. Since there are many sub-loops inside the function, it is impossible to directly use the "pipeline rewind" pragma, so you need to try to use different algorithms to achieve it.

Overall, this project taught me how Transformers’ attention can be mapped to a **systolic array** approach in HLS, how **XRT** can manage device buffers, and how to unify these steps into a working flow for FPGA acceleration.
