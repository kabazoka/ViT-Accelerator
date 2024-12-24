# ViT-accleration
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
The Python host code was originally authored by https://github.com/jankrepl and was subsequently copied from his repository: https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/vision_transformer. All credit for this code is due to the original author.
