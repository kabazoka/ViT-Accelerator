open_project hls_attention
add_files ./kernel.cpp
add_files ./q_matmul_k_function.cpp
add_files ./testbench.cpp -tb
open_solution "solution1"
set_part {xcu50-fsvh2104-2-e}

# Define the clock
create_clock -period 10 -name default

set_top attention_kernel

# Import configurations
source ./scripts/synth_config.tcl

csim_design
csynth_design
cosim_design
export_design -rtl verilog -format ip_catalog