# ViT-Accelerator
Vision Transformer Accelerator implemented in Vivado HLS for Xilinx FPGAs.

## Requirements
- python=3.11
  - Library dependencies are listed in host/requirements.txt
- Xilinx Vitis Toolkit=2022.1

## Usage
```bash
pip install -r host/requirements.txt
python host/custom.py
python host/verify.py
python host/forward.py
cd hls_source
vitis_hls -f scripts/run_hls.tcl
```

## Declaration
The C++ host code was originally authored by https://github.com/staghado and was subsequently copied from his repository: https://github.com/staghado/vit.cpp. All credit for this code is due to the original author.
