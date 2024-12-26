# ViT-Accelerator
Vision Transformer Accelerator implemented in Vivado HLS for Xilinx FPGAs.

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
