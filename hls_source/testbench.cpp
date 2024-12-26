#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iomanip>  // 為了印出浮點數時更容易觀察
#include "kernel.h"

// 與 kernel.cpp 相同的定義
#define SCAILING_FACTOR 8
#define REAL_Q_ROW 197
#define REAL_Q_COL 64
#define REAL_K_ROW 64
#define REAL_K_COL 197

// 與 kernel.cpp 相同的近似 exponential 函式
float exp_approx_ref(float x) {
    // 這個近似是 kernel.cpp 中給的簡單級數展開
    // e^x ≈ 1 + x + x^2/2 + x^3/6
    return 1.0f + x + (x * x) / 2.0f + (x * x * x) / 6.0f;
}

// 參考用的軟體實作: attention = softmax( (Q*K) / SCAILING_FACTOR )
void reference_attention(const float* Q, const float* K, float* ref_attention)
{
    // 1) 先做矩陣乘法： Q * K => score
    //    Q shape: (197 x 64), K shape: (64 x 197), score shape: (197 x 197)
    static float score[REAL_Q_ROW][REAL_K_COL];

    // 初始化
    for(int i = 0; i < REAL_Q_ROW; i++){
        for(int j = 0; j < REAL_K_COL; j++){
            score[i][j] = 0.0f;
        }
    }

    // Q*K
    for(int i = 0; i < REAL_Q_ROW; i++){
        for(int j = 0; j < REAL_K_COL; j++){
            float sum_val = 0.0f;
            for(int k = 0; k < REAL_Q_COL; k++){
                // Q[i,k], K[k,j]
                sum_val += Q[i * REAL_Q_COL + k] * K[k * REAL_K_COL + j];
            }
            score[i][j] = sum_val;
        }
    }

    // 2) scaling => score[i][j] /= SCAILING_FACTOR
    for(int i = 0; i < REAL_Q_ROW; i++){
        for(int j = 0; j < REAL_K_COL; j++){
            score[i][j] /= (float)SCAILING_FACTOR;
        }
    }

    // 3) softmax(使用相同的 exp_approx)
    //    這裡注意 softmax 的計算是對 row 做 (對行內所有元素 exponent 後再normalize)
    //    kernel.cpp 中做法：每一 row 裏面先算 exponent, 再把 exponent 的和 sum[0] 做分母
    static float softmax_scores[REAL_Q_ROW][REAL_K_COL];
    for(int i = 0; i < REAL_Q_ROW; i++){
        // 先把 exponent 做部分累加, kernel 裏面是用 sum[8] 累加
        // 這裡簡單起見，直接做單一 sum
        float sum_exp = 0.0f;
        for(int j = 0; j < REAL_K_COL; j++){
            float e_val = exp_approx_ref(score[i][j]);
            sum_exp += e_val;
            softmax_scores[i][j] = e_val;  // 暫存 e^x
        }
        // normalize
        for(int j = 0; j < REAL_K_COL; j++){
            softmax_scores[i][j] = softmax_scores[i][j] / sum_exp;
        }
    }

    // 4) 回寫到 ref_attention (flatten)
    for(int i = 0; i < REAL_Q_ROW; i++){
        for(int j = 0; j < REAL_K_COL; j++){
            ref_attention[i * REAL_K_COL + j] = softmax_scores[i][j];
        }
    }
}

int main()
{
    // 建立輸入與輸出緩衝
    static float q[REAL_Q_ROW * REAL_Q_COL];
    static float k[REAL_K_ROW * REAL_K_COL];
    static float attention_score[REAL_Q_ROW * REAL_K_COL];      // 硬體函式輸出
    static float reference_score[REAL_Q_ROW * REAL_K_COL];      // 軟體參考結果

    // 1) 初始化 Q, K (可改用隨機或指定)
    for(int i = 0; i < REAL_Q_ROW * REAL_Q_COL; i++){
        q[i] = float(std::rand() % 100) / 10.0f;   // 0~9.9
    }
    for(int i = 0; i < REAL_K_ROW * REAL_K_COL; i++){
        k[i] = float(std::rand() % 100) / 10.0f;   // 0~9.9
    }

    // 2) 計算參考結果 (純軟體)
    reference_attention(q, k, reference_score);

    // 3) 呼叫硬體函式 attention_kernel
    //    因為函式宣告是 (volatile float* q, volatile float* k, volatile float* attention_score)
    //    這邊可以直接把靜態陣列轉成 volatile float* 傳入
    attention_kernel(q, k, attention_score);

    // 4) 驗證結果 (比對 reference_score 與 attention_score)
    //    由於內部包含浮點運算與近似 exponent，建議允許一定的相對或絕對誤差
    bool pass = true;
    double max_diff = 0.0;
    for(int i = 0; i < REAL_Q_ROW * REAL_K_COL; i++){
        float ref_val = reference_score[i];
        float hw_val  = attention_score[i];
        // 計算絕對誤差
        float diff = std::fabs(ref_val - hw_val);
        if(diff > 2e-2) {
            pass = false;
        }
        if(diff > max_diff) {
            max_diff = diff;
        }
    }

    // 5) 印出測試結果
    if(pass){
        std::cout << "===================================\n";
        std::cout << " Test Passed. Max diff = " << max_diff << "\n";
        std::cout << "===================================\n";
    } else {
        std::cout << "===================================\n";
        std::cout << " Test Failed. Max diff = " << max_diff << "\n";
        std::cout << "===================================\n";
    }

    return (pass ? 0 : 1);
}
