#include <hls_math.h>
// #define D_HEAD 64
// #define NUM_TOKENS 128
// Changed small values for testing
#define D_HEAD 16
#define NUM_TOKENS 32

float exp_approx(float x) {
    float result = 1 + x + (x * x) / 2 + (x * x * x) / 6;
    return result;
}

extern "C" {
void attention_kernel(float Q[NUM_TOKENS][D_HEAD],
                      float K[NUM_TOKENS][D_HEAD],
                      float V[NUM_TOKENS][D_HEAD],
                      float Output[NUM_TOKENS][D_HEAD]) {
#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=K offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=V offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=Output offset=slave bundle=gmem
#pragma HLS ARRAY_PARTITION variable=Q cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=K cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=V cyclic factor=4 dim=2
#pragma HLS INTERFACE s_axilite port=Output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    float scores[NUM_TOKENS][NUM_TOKENS];
    float softmax_scores[NUM_TOKENS][NUM_TOKENS];
    
    // Compute scaled dot product (Query-Key)
    for (int i = 0; i < NUM_TOKENS; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < NUM_TOKENS; j++) {
            float dot_product = 0;
            for (int k = 0; k < D_HEAD; k++) {
                #pragma HLS UNROLL factor=4
                dot_product += Q[i][k] * K[j][k];
            }
            scores[i][j] = dot_product / hls::sqrt((float)D_HEAD);
        }
    }

    // Compute softmax (approximated for simplicity)
    for (int i = 0; i < NUM_TOKENS; i++) {
        float sum = 0;
        for (int j = 0; j < NUM_TOKENS; j++) {
            scores[i][j] = exp_approx(scores[i][j]);
            sum += scores[i][j];
        }
        for (int j = 0; j < NUM_TOKENS; j++) {
            softmax_scores[i][j] = scores[i][j] / sum;
        }
    }

    // Attention-Value Dot Product
    for (int i = 0; i < NUM_TOKENS; i++) {
        for (int j = 0; j < D_HEAD; j++) {
            float result = 0;
            for (int k = 0; k < NUM_TOKENS; k++) {
                result += softmax_scores[i][k] * V[k][j];
            }
            Output[i][j] = result;
        }
    }
}
}