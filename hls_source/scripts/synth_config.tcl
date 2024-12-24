config_compile -pipeline_loops 1
config_rtl -reset all
set_directive_pipeline -II 1 "attention_kernel"
set_directive_array_partition -type cyclic -factor 4 -dim 2 "attention_kernel" Q
set_directive_array_partition -type cyclic -factor 4 -dim 2 "attention_kernel" K
set_directive_array_partition -type cyclic -factor 4 -dim 2 "attention_kernel" V
set_directive_array_partition -type cyclic -factor 4 -dim 2 "attention_kernel" Output