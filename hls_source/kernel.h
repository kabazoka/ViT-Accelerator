#ifndef KERNEL_H
#define KERNEL_H

#define D_HEAD 64         // Dimension of each head
#define NUM_TOKENS 128    // Number of tokens (sequence length)

// Function prototype for the HLS kernel
extern "C" {
void attention_kernel(float Q[NUM_TOKENS][D_HEAD],
                      float K[NUM_TOKENS][D_HEAD],
                      float V[NUM_TOKENS][D_HEAD],
                      float Output[NUM_TOKENS][D_HEAD]);
}

#endif // KERNEL_H