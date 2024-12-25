#include <hls_math.h>
// #define D_HEAD 64
// #define NUM_TOKENS 128
// Changed small values for testing
#define D_HEAD 16
#define NUM_TOKENS 32

extern "C" {
void attention_kernel(float Q[NUM_TOKENS][D_HEAD],
                      float K[NUM_TOKENS][D_HEAD],
                      float V[NUM_TOKENS][D_HEAD],
                      float Output[NUM_TOKENS][D_HEAD]) {

    float scores[NUM_TOKENS][NUM_TOKENS];
    float softmax_scores[NUM_TOKENS][NUM_TOKENS];
    
    // Compute scaled dot product (Query-Key)
    for (int i = 0; i < NUM_TOKENS; i++) {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < NUM_TOKENS; j++) {
            float dot_product = 0;
            for (int k = 0; k < D_HEAD; k++) {
                dot_product += Q[i][k] * K[j][k];
            }
            scores[i][j] = dot_product / hls::sqrt((float)D_HEAD);
        }
    }

    // Compute softmax (approximated for simplicity)
    for (int i = 0; i < NUM_TOKENS; i++) {
    	float sum[8] = {0};
        for (int j = 0; j < NUM_TOKENS; j++) {
#pragma HLS PIPELINE II=1
			sum[j%8] += hls::exp(scores[i][j]);
        }
        for (int j = 1; j < 8; j++) {
#pragma HLS PIPELINE II=7
        	sum[0] += sum[j];
        }
        for (int j = 0; j < NUM_TOKENS; j++) {
#pragma HLS PIPELINE II=1
			softmax_scores[i][j] = scores[i][j] / sum[0];
		}
    }

    // Attention-Value Dot Product
    for (int i = 0; i < NUM_TOKENS; i++) {
        for (int j = 0; j < D_HEAD; j++) {
#pragma HLS PIPELINE II=1
            float result = 0;
            for (int k = 0; k < NUM_TOKENS; k++) {
                result += softmax_scores[i][k] * V[k][j];
            }
            Output[i][j] = result;
        }
    }
}
}
