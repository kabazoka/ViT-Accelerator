#define _CRT_SECURE_NO_DEPRECATE // disables "unsafe" warnings on Windows

#include "vit.h"
// Only include XRT headers if we're using FPGA:
#ifdef USE_FPGA
  #include "xrt.h"
  #include "xrt/experimental/xrt_kernel.h"
#endif

// for ggml
#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/examples/stb_image.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <thread>
#include <cinttypes>
#include <algorithm>
#include <ctime>

// main
int main(int argc, char **argv) {
    const int64_t t_main_start_us = ggml_time_us();

    // 1) Parse command-line arguments
    vit_params params;
    if (!vit_params_parse(argc, argv, params)) {
        return 1;
    }
    if (params.seed < 0) {
        params.seed = time(NULL);
    }
    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);
    fprintf(stderr, "%s: n_threads = %d / %d\n",
            __func__, params.n_threads, (int32_t)std::thread::hardware_concurrency());

    // 2) Create model + state
    vit_model model;
    vit_state state;
    // We'll set up model + state whether or not we have FPGA

#ifdef USE_FPGA
    // -------------------------------------------------------------
    // 2a) If FPGA is enabled, create device, load xclbin, kernel
    // -------------------------------------------------------------
    std::string xclbinFilename = "kernel.xclbin"; // adjust path as needed
    try {
        xrt::device device(0); // open device #0
        auto uuid = device.load_xclbin(xclbinFilename);
        fprintf(stderr, "Loaded xclbin: %s\n", xclbinFilename.c_str());

        // The kernel name here must match your HLS code
        xrt::kernel krnl_attention = xrt::kernel(device, uuid, "attention_kernel");

        // Store kernel handle in 'state' so vit.cpp can call it
        state.krnl_attention = krnl_attention;
    }
    catch (const std::exception &ex) {
        fprintf(stderr, "ERROR loading xclbin or creating kernel: %s\n", ex.what());
        return 1;
    }
#endif // USE_FPGA

    // 3) Load ViT model
    int64_t t_load_us = 0;
    {
        const int64_t t_start_us = ggml_time_us();
        if (!vit_model_load(params.model.c_str(), model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n",
                    __func__, params.model.c_str());
            return 1;
        }
        t_load_us = ggml_time_us() - t_start_us;
    }

    // 4) Load input image
    image_u8 img0;
    if (!load_image_from_file(params.fname_inp.c_str(), img0)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n",
                __func__, params.fname_inp.c_str());
        return 1;
    }
    fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n",
            __func__, params.fname_inp.c_str(), img0.nx, img0.ny);

    // 5) Preprocess image
    image_f32 img1;
    if (vit_image_preprocess(img0, img1, model.hparams)) {
        fprintf(stderr, "processed, out dims : (%d x %d)\n", img1.nx, img1.ny);
    } else {
        fprintf(stderr, "Failed to preprocess image. Possibly unknown interpolation mode?\n");
        return 1;
    }

    // 6) Initialize GGML context for inference
    {
        static size_t buf_size = 3u * 1024 * 1024; // e.g. 3 MB
        struct ggml_init_params ggml_params = {
            buf_size, // mem_size
            NULL,     // mem_buffer
            false     // no_alloc
        };
        state.ctx = ggml_init(ggml_params);
        state.prediction = ggml_new_tensor_4d(
            state.ctx, GGML_TYPE_F32, model.hparams.num_classes, 1, 1, 1);
    }

    // 7) Run the inference
    std::vector<std::pair<float, int>> predictions;
    {
        int ret = vit_predict(model, state, img1, params, predictions);
        if (ret != 0) {
            fprintf(stderr, "%s: vit_predict failed\n", __func__);
            return 1;
        }
    }

    // 8) Timing & cleanup
    {
        const int64_t t_main_end_us = ggml_time_us();
        fprintf(stderr, "\n\n");
        fprintf(stderr, "%s:    model load time = %8.2f ms\n",
                __func__, t_load_us / 1000.0f);
        fprintf(stderr, "%s:    processing time = %8.2f ms\n",
                __func__, (t_main_end_us - t_main_start_us - t_load_us) / 1000.0f);
        fprintf(stderr, "%s:    total time      = %8.2f ms\n",
                __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    ggml_free(model.ctx);
    return 0;
}
