
#include <iostream>
#include <cstdlib>
#include "mmult.h"

using namespace std;

#define A_ROW 197
#define A_COL 64
#define B_ROW 64
#define B_COL 197

void reference_mmult(float* a, float* b, float* c, int a_row, int a_col, int b_col) {
    for (int i = 0; i < a_row; i++) {
        for (int j = 0; j < b_col; j++) {
            float sum = 0.0f;
            for (int k = 0; k < a_col; k++) {
                sum += a[i * a_col + k] * b[k * b_col + j];
            }
            c[i * b_col + j] = sum;
        }
    }
}

void print_matrix(float* matrix, size_t rows, size_t cols, const char* name) {
    std::cout << "Matrix " << name << ":\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void mmult(volatile float* a, volatile float* b, volatile float* c);

int main(int argc, char *argv[]) {
	float a[A_ROW * A_COL];
	float b[B_ROW * B_COL];
	float c[A_ROW * B_COL];
	float c_ref[A_ROW * B_COL];

    // Initialize matrices with random data
    for (int i = 0; i < A_ROW * A_COL; i++) {
        a[i] = static_cast<float>(std::rand()) / RAND_MAX * 10.0f;
    }
    for (int i = 0; i < B_ROW * B_COL; i++) {
        b[i] = static_cast<float>(std::rand()) / RAND_MAX * 10.0f;
    }

    // Print the initialized matrices
    print_matrix(a, A_ROW, A_COL, "A");
    print_matrix(b, B_ROW, B_COL, "B");

    // Compute the reference result
    reference_mmult(a, b, c_ref, A_ROW, A_COL, B_COL);

    // Run the HLS kernel
    mmult(a, b, c);

    // Compare results
    bool pass = true;
    for (int i = 0; i < A_ROW * B_COL; i++) {
        if (std::abs(c[i] - c_ref[i]) > 1e-5) { // Tolerance for floating-point comparison
            pass = false;
            std::cout << "Mismatch at index " << i << ": HW=" << c[i] << ", Ref=" << c_ref[i] << "\n";
        }
    }

    if (pass) {
        std::cout << "Test Passed\n";
    } else {
        std::cout << "Test Failed\n";
    }

    return pass ? 0 : 1;
}

