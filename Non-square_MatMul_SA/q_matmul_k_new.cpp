#include <iostream>

// ---------------------- SIZES & DEFINES ----------------------
// Original parameters from the problem
#define REAL_A_ROW 197
#define REAL_A_COL 64
#define REAL_B_ROW 64
#define REAL_B_COL 197

// Padding Size
#define PAD_A_ROW 208 // 197 + (16 - 5) = 208
#define PAD_B_COL 208

// Size of each tile
#define M 16

// ------------------ TOP FUNCTION (KERNEL) --------------------
extern "C" {
void mmult(volatile float* a, // Read-Only Matrix A
           volatile float* b, // Read-Only Matrix B
           volatile float* c  // Output Result
          )
{
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0 depth=12608
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1 depth=12608
#pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2 depth=38809

#pragma HLS INTERFACE s_axilite port=a bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=c bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // ------------------------------------------------------------
    // 1) Declare three large static arrays (localA, localB, localC)
    //    to store the entire input/output.
    //    Use pragma to force them to be inferred as BRAM.
    // ------------------------------------------------------------
    static float localA[REAL_A_ROW][REAL_A_COL];
#pragma HLS BIND_STORAGE variable=localA type=RAM_2P impl=BRAM latency=2
    static float localB[REAL_B_ROW][REAL_B_COL];
#pragma HLS BIND_STORAGE variable=localB type=RAM_2P impl=BRAM latency=2
    static float localC[REAL_A_ROW][REAL_B_COL];
#pragma HLS BIND_STORAGE variable=localC type=RAM_2P impl=BRAM latency=2

    // ------------------------------------------------------------
    // 2) Use a single burst read to load A and B from global memory
    //    into localA and localB (197×64, 64×197).
    // ------------------------------------------------------------
    // Read Input A => localA
readA:
    for (int loc = 0, i = 0, j = 0; loc < (REAL_A_ROW * REAL_A_COL); loc++, j++) {
#pragma HLS PIPELINE II=1
        if (j == REAL_A_COL) {
            i++;
            j = 0;
        }
        localA[i][j] = a[loc];
    }

    // Read Input B => localB
readB:
    for (int loc = 0, i = 0, j = 0; loc < (REAL_B_ROW * REAL_B_COL); loc++, j++) {
#pragma HLS PIPELINE II=1
        if (j == REAL_B_COL) {
            i++;
            j = 0;
        }
        localB[i][j] = b[loc];
    }

    // ------------------------------------------------------------
    // 3) Initialize localC (to prevent old data from interfering),
    //    because the final result will be stored here.
    // ------------------------------------------------------------
initC:
    for (int i = 0; i < REAL_A_ROW; i++) {
        for (int j = 0; j < REAL_B_COL; j++) {
#pragma HLS PIPELINE II=1
            localC[i][j] = 0.0f;
        }
    }

    // ------------------------------------------------------------
    // The following starts the Systolic Array computation (tile-based):
    //  - Since we've already read the entire A and B into localA, localB,
    //    we only need to access localA, localB during the computation.
    //  - Finally, write the calculated localC back to global memory.
    // ------------------------------------------------------------

    // Use subA, subB as temporary buffers for each tile from localA, localB
    float subA[M][REAL_A_COL];
#pragma HLS ARRAY_PARTITION variable=subA dim=2 complete

    float subB[REAL_B_ROW][M];
#pragma HLS ARRAY_PARTITION variable=subB dim=1 complete

    // Use inC to replace the previous localC for storing
    // intermediate results for a 16×16 tile
    float inC[M][M];
#pragma HLS ARRAY_PARTITION variable=inC dim=0 complete

    // Number of tiles: 208/16 = 13
tile_outer_loop:
    for (int tileRow = 0; tileRow < (PAD_A_ROW / M); tileRow++) {   // 0..12
        for (int tileCol = 0; tileCol < (PAD_B_COL / M); tileCol++) { // 0..12

            //------------------------------------------------------
            // (a) Initialize inC to 0 before starting the tile computation
            //------------------------------------------------------
        init_inC:
            for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE
                for (int j = 0; j < M; j++) {
                    inC[i][j] = 0.0f;
                }
            }

            //------------------------------------------------------
            // (b) Retrieve the sub-block of data needed for this tile
            //     from localA, localB into subA, subB.
            //     Perform zero-padding if out of valid range.
            //------------------------------------------------------
        read_subA:
            for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE
                for (int j = 0; j < REAL_A_COL; j++) { // j < 64
                    int globalRow = tileRow * M + i;  // 0..207
                    // real col = j
                    if (globalRow < REAL_A_ROW) {  // <197
                        subA[i][j] = localA[globalRow][j];
                    } else {
                        subA[i][j] = 0.0f;
                    }
                }
            }

        read_subB:
            for (int i = 0; i < REAL_B_ROW; i++) {    // i < 64
#pragma HLS PIPELINE
                for (int j = 0; j < M; j++) {        // j < 16
                    int globalCol = tileCol * M + j; // 0..207
                    // real row = i
                    if (globalCol < REAL_B_COL) {   // <197
                        subB[i][j] = localB[i][globalCol];
                    } else {
                        subB[i][j] = 0.0f;
                    }
                }
            }

            //------------------------------------------------------
            // (c) Internal Systolic Array computation:
            //     subA is 16×64, subB is 64×16
            //     The result goes into inC (16×16).
            //------------------------------------------------------
        systolic_tiling:
            // inA, inB are additional scratch registers inside the PE array
            float inA[M][M];
#pragma HLS ARRAY_PARTITION variable=inA dim=0 complete
            float inB[M][M];
#pragma HLS ARRAY_PARTITION variable=inB dim=0 complete

        init_inAB:
            for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE
                for (int j = 0; j < M; j++) {
                    inA[i][j] = 0.0f;
                    inB[i][j] = 0.0f;
                }
            }

            // Total shift steps = 64 + 2*16 - 2 = 64 + 30 = 94
            for (int r = 0; r < (REAL_A_COL + 2 * M - 2); r++) {
#pragma HLS PIPELINE
                // 1) Shift inA to the right
                for (int i = 0; i < M; i++) {
                    for (int j = M - 1; j >= 1; j--) {
                        inA[i][j] = inA[i][j - 1];
                    }
                }

                // 2) Shift inB downward
                for (int i = M - 1; i >= 1; i--) {
                    for (int j = 0; j < M; j++) {
                        inB[i][j] = inB[i - 1][j];
                    }
                }

                // 3) Inject new data into the leftmost column of inA
                //    and the top row of inB
                for (int i = 0; i < M; i++) {
                    int global_col = r - i;
                    if ((global_col >= 0) && (global_col < REAL_A_COL)) {
                        inA[i][0] = subA[i][global_col];
                    } else {
                        inA[i][0] = 0.0f;
                    }
                }
                for (int j = 0; j < M; j++) {
                    int global_row = r - j;
                    if ((global_row >= 0) && (global_row < REAL_B_ROW)) {
                        inB[0][j] = subB[global_row][j];
                    } else {
                        inB[0][j] = 0.0f;
                    }
                }

                // 4) PE calculation: inC[i][j] += inA[i][j] * inB[i][j]
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < M; j++) {
                        inC[i][j] += inA[i][j] * inB[i][j];
                    }
                }
            } // end of systolic loop

            //------------------------------------------------------
            // (d) Write the tile's computed result (inC) back to
            //     the corresponding location in localC,
            //     only if it's within the valid 197×197 range.
            //------------------------------------------------------
        store_tileC:
#pragma HLS DEPENDENCE variable=localC inter false
            for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE II=1
                for (int j = 0; j < M; j++) {
                    int globalRow = tileRow * M + i; // 0..207
                    int globalCol = tileCol * M + j; // 0..207
                    // Only write back if within the valid region of 197×197
                    if ((globalRow < REAL_A_ROW) && (globalCol < REAL_B_COL)) {
                        // Use += as in the original code
                        localC[globalRow][globalCol] += inC[i][j];
                    }
                }
            }

        } // tileCol
    } // tileRow

    // ------------------------------------------------------------
    // 4) After computation finishes, write localC back to global
    //    memory c with a single burst write (197×197).
    // ------------------------------------------------------------
writeC:
    for (int loc = 0, i = 0, j = 0; loc < (REAL_A_ROW * REAL_B_COL); loc++, j++) {
#pragma HLS PIPELINE II=1
        if (j == REAL_B_COL) {
            i++;
            j = 0;
        }
        c[loc] = localC[i][j];
    }
}
} // extern "C"
