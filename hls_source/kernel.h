#ifndef KERNEL_H
#define KERNEL_H

#define REAL_Q_ROW 197
#define REAL_Q_COL 64
#define REAL_K_ROW 64
#define REAL_K_COL 197

extern "C" {
void mmult(float A[REAL_Q_ROW][REAL_Q_COL], float B[REAL_K_ROW][REAL_K_COL], float C[REAL_Q_ROW][REAL_K_COL]);
}

extern "C" {
void attention_kernel(volatile float* q, volatile float* k, volatile float* attention_score);
}

#endif // KERNEL_H
