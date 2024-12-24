
# ifndef __MMULT_H__
# define __MMULT_H__

extern "C"{
void mmult(volatile int* a, volatile int* b, volatile int* c, int a_row, int a_col, int b_row, int b_col);
}

#endif
