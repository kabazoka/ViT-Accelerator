// kernel.cpp
#include <iostream>
#include "kernel.h"

#define SCAILING_FACTOR 8
#define REAL_Q_ROW 197
#define REAL_Q_COL 64
#define REAL_K_ROW 64
#define REAL_K_COL 197

float exp_approx(float x) {
    float result = 1 + x + (x * x) / 2 + (x * x * x) / 6;
    return result;
}

extern "C" {
void mmult(float A[REAL_Q_ROW][REAL_Q_COL], float B[REAL_K_ROW][REAL_K_COL], float C[REAL_Q_ROW][REAL_K_COL]);
}

extern "C" {
void attention_kernel(volatile float* q, // Read-Only Matrix Q
                      volatile float* k, // Read-Only Matrix K
                      volatile float* attention_score  // Output Result
                     ) {

#pragma HLS INTERFACE m_axi port=q offset=slave bundle=gmem0 depth=12608
#pragma HLS INTERFACE m_axi port=k offset=slave bundle=gmem1 depth=12608
#pragma HLS INTERFACE m_axi port=attention_score offset=slave bundle=gmem2 depth=38809

#pragma HLS INTERFACE s_axilite port=q bundle=control
#pragma HLS INTERFACE s_axilite port=k bundle=control
#pragma HLS INTERFACE s_axilite port=attention_score bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float localQ[REAL_Q_ROW][REAL_Q_COL];
#pragma HLS BIND_STORAGE variable=localQ type=RAM_2P impl=BRAM latency=2
    static float localK[REAL_K_ROW][REAL_K_COL];
#pragma HLS BIND_STORAGE variable=localK type=RAM_2P impl=BRAM latency=2
    static float localAttention[REAL_Q_ROW][REAL_K_COL];
#pragma HLS BIND_STORAGE variable=localAttention type=RAM_2P impl=BRAM latency=2

    static float scores[REAL_Q_ROW][REAL_K_COL];
    static float softmax_scores[REAL_Q_ROW][REAL_K_COL];

    // Read Input q => localQ
readQ:
    for (int loc = 0, i = 0, j = 0; loc < (REAL_Q_ROW * REAL_Q_COL); loc++, j++) {
#pragma HLS PIPELINE II=1
        if (j == REAL_Q_COL) {
            i++;
            j = 0;
        }
        localQ[i][j] = q[loc];
    }

    // Read Input k => localK
readK:
    for (int loc = 0, i = 0, j = 0; loc < (REAL_K_ROW * REAL_K_COL); loc++, j++) {
#pragma HLS PIPELINE II=1
        if (j == REAL_K_COL) {
            i++;
            j = 0;
        }
        localK[i][j] = k[loc];
    }
    // Initialize localAttention
init_scores:
    for (int i = 0; i < REAL_Q_ROW; i++) {
        for (int j = 0; j < REAL_K_COL; j++) {
#pragma HLS PIPELINE II=1
            scores[i][j] = 0.0f;
        }
    }
    // query matmul  key
    mmult(localQ, localK, scores);

    // Scaling query & key dot-product
scailing:
    for (int i = 0; i < REAL_Q_ROW; i++) {
        for (int j = 0; j < REAL_K_COL; j++){
#pragma HLS PIPELINE II=1
            scores[i][j] = scores[i][j] / (float)SCAILING_FACTOR ;
        }
    }

    // Compute softmax (approximated for simplicity)
softmax:
    for (int i = 0; i < REAL_Q_ROW; i++) {
        float sum[8] = {0};
        for (int j = 0; j < REAL_K_COL; j++) {
#pragma HLS PIPELINE II=1
            sum[j % 8] += exp_approx(scores[i][j]);
        }
        for (int j = 1; j < 8; j++) {
#pragma HLS PIPELINE II=7
            sum[0] += sum[j];
        }
        for (int j = 0; j < REAL_K_COL; j++) {
#pragma HLS PIPELINE II=1
			softmax_scores[i][j] = scores[i][j] / sum[0];
		}
	}

writeAttention:
    for (int loc = 0, i = 0, j = 0; loc < (REAL_Q_ROW * REAL_K_COL); loc++, j++) {
#pragma HLS PIPELINE II=1
        if (j == REAL_K_COL) {
            i++;
            j = 0;
        }
        attention_score[loc] = softmax_scores[i][j];
    }

}
}
