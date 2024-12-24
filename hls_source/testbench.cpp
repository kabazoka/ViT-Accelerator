#include <iostream>
#include "kernel.h"

#define D_HEAD 64
#define NUM_TOKENS 128

void initialize_data(float Q[NUM_TOKENS][D_HEAD], float K[NUM_TOKENS][D_HEAD], float V[NUM_TOKENS][D_HEAD]) {
    for (int i = 0; i < NUM_TOKENS; i++) {
        for (int j = 0; j < D_HEAD; j++) {
            Q[i][j] = (float)(i + j) / 100;
            K[i][j] = (float)(i - j) / 100;
            V[i][j] = (float)(i * j) / 1000;
        }
    }
}

int main() {
    float Q[NUM_TOKENS][D_HEAD];
    float K[NUM_TOKENS][D_HEAD];
    float V[NUM_TOKENS][D_HEAD];
    float Output[NUM_TOKENS][D_HEAD] = {0};

    initialize_data(Q, K, V);

    // Run kernel
    attention_kernel(Q, K, V, Output);

    // Print result
    for (int i = 0; i < 10; i++) { // Print first 10 results
        for (int j = 0; j < 10; j++) {
            std::cout << Output[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}